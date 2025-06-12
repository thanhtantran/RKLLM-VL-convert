from data.make_input_embeds_for_quantize import MakeInputEmbeds
import numpy as np
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from rknn.api import RKNN
from rkllm.api import RKLLM
from tqdm import tqdm
import torch
import argparse
import sys
import gc

def get_gpu_memory_info():
  """Get GPU memory information"""
  if torch.cuda.is_available():
      total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
      allocated = torch.cuda.memory_allocated() / 1024**3
      reserved = torch.cuda.memory_reserved() / 1024**3
      free = total_memory - reserved
      return total_memory, allocated, reserved, free
  return 0, 0, 0, 0

def clear_memory():
  """Aggressive memory cleanup"""
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.synchronize()

def get_user_preferences():
  """Get user preferences for device and precision"""
  print("=== Model Conversion Configuration ===")
  
  # Check GPU memory first
  cuda_available = torch.cuda.is_available()
  if cuda_available:
      total_mem, allocated, reserved, free = get_gpu_memory_info()
      print(f"GPU: {torch.cuda.get_device_name(0)}")
      print(f"Total VRAM: {total_mem:.1f}GB, Available: {free:.1f}GB")
      
      if total_mem < 6:  # Less than 6GB
          print("⚠️  Warning: Limited VRAM detected. Consider using CPU or aggressive optimization.")
      
      print("Device options:")
      print("1. GPU (CUDA) - Faster but needs more memory")
      print("2. CPU - Slower but more reliable for large models")
      print("3. GPU with aggressive memory optimization")
      
      while True:
          choice = input("Choose device (1/2/3): ").strip()
          if choice == '1':
              device_choice = 'gpu'
              memory_aggressive = False
              break
          elif choice == '2':
              device_choice = 'cpu'
              memory_aggressive = False
              break
          elif choice == '3':
              device_choice = 'gpu'
              memory_aggressive = True
              break
          else:
              print("Please enter 1, 2, or 3")
  else:
      print("CUDA is not available. Using CPU.")
      device_choice = 'cpu'
      memory_aggressive = False
  
  # Precision selection
  if device_choice == 'gpu':
      print("\nPrecision options:")
      print("1. Float32 - Higher precision, more memory")
      print("2. Float16 - Less memory, faster (Recommended for limited VRAM)")
      print("3. Int8 - Minimal memory (Experimental)")
      
      while True:
          choice = input("Choose precision (1/2/3): ").strip()
          if choice == '1':
              precision_choice = 'float32'
              break
          elif choice == '2':
              precision_choice = 'float16'
              break
          elif choice == '3':
              precision_choice = 'int8'
              break
          else:
              print("Please enter 1, 2, or 3")
  else:
      precision_choice = 'float32'
  
  return device_choice, precision_choice, memory_aggressive

def setup_device_and_precision(device_choice, precision_choice, memory_aggressive):
  """Setup device and precision based on user choice"""
  if device_choice == 'gpu' and torch.cuda.is_available():
      device = torch.device("cuda")
      print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
  else:
      device = torch.device("cpu")
      print(f"\n✓ Using CPU")
  
  # Set precision
  if precision_choice == 'float16':
      model_dtype = torch.float16
      tensor_dtype = torch.float16
      print(f"✓ Using Float16 precision")
  elif precision_choice == 'int8':
      model_dtype = torch.int8
      tensor_dtype = torch.float32  # Keep input as float32
      print(f"✓ Using Int8 precision (Experimental)")
  else:
      model_dtype = torch.float32
      tensor_dtype = torch.float32
      print(f"✓ Using Float32 precision")
  
  if memory_aggressive:
      print(f"✓ Aggressive memory optimization enabled")
  
  return device, model_dtype, tensor_dtype

def load_model_with_memory_optimization(quantize, device, model_dtype, memory_aggressive):
  """Load model with aggressive memory optimization"""
  print(f"\n📥 Loading model with memory optimization...")
  
  # Clear memory before loading
  clear_memory()
  
  load_kwargs = {
      'torch_dtype': model_dtype,
      'trust_remote_code': True,
      'low_cpu_mem_usage': True,
  }
  
  if memory_aggressive and device.type == 'cuda':
      # Use CPU offloading for large models
      load_kwargs['device_map'] = 'auto'
      load_kwargs['max_memory'] = {0: "6GiB", "cpu": "16GiB"}  # Limit GPU usage
      print("Using CPU offloading for memory optimization")
  
  try:
      model = Qwen2VLForConditionalGeneration.from_pretrained(
          quantize.path, **load_kwargs
      ).eval()
      
      tokenizer = AutoTokenizer.from_pretrained(
          quantize.path, trust_remote_code=True, use_fast=False
      )
      
      # Move to device if not using device_map
      if not memory_aggressive or device.type == 'cpu':
          model = model.to(device)
      
      print(f"✓ Model loaded successfully")
      
      # Print memory usage
      if device.type == 'cuda':
          total_mem, allocated, reserved, free = get_gpu_memory_info()
          print(f"GPU Memory after loading: {allocated:.1f}GB used, {free:.1f}GB free")
      
      return model, tokenizer
      
  except RuntimeError as e:
      if "out of memory" in str(e).lower():
          print(f"❌ GPU out of memory during model loading")
          print(f"Falling back to CPU...")
          clear_memory()
          
          # Fallback to CPU
          load_kwargs_cpu = {
              'torch_dtype': torch.float32,
              'trust_remote_code': True,
              'low_cpu_mem_usage': True,
          }
          
          model = Qwen2VLForConditionalGeneration.from_pretrained(
              quantize.path, **load_kwargs_cpu
          ).eval()
          
          tokenizer = AutoTokenizer.from_pretrained(
              quantize.path, trust_remote_code=True, use_fast=False
          )
          
          print(f"✓ Model loaded on CPU as fallback")
          return model, tokenizer
      else:
          raise e

def process_with_checkpointing(model, input_tensor, device, memory_aggressive):
  """Process model with gradient checkpointing and memory management"""
  
  def export_onnx(image):
      # Ensure image is on the correct device
      image = image.to(device)
      
      # Process in smaller chunks if memory aggressive
      if memory_aggressive and device.type == 'cuda':
          # Clear cache before processing
          clear_memory()
      
      patches = image.repeat(2, 1, 1, 1)  # temporal_patch_size
      patches = patches.reshape(1, 2, 3, 14, 2, 14, 14, 2, 14)  # Adjusted dimensions
      patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
      flatten_patches = patches.reshape(784, 1176)  # 28*28, 3*2*14*14
      
      model.visual.forward = forward_new(model.visual)
      
      try:
          with torch.no_grad():
              # Use gradient checkpointing if available
              if hasattr(model.visual, 'gradient_checkpointing_enable'):
                  model.visual.gradient_checkpointing_enable()
              
              feature = model.visual(flatten_patches)
              
              # Clear intermediate results
              if memory_aggressive and device.type == 'cuda':
                  clear_memory()
              
              return feature
              
      except RuntimeError as e:
          if "out of memory" in str(e).lower():
              print(f"❌ GPU OOM during processing. Trying CPU fallback...")
              clear_memory()
              
              # Move to CPU for processing
              model_cpu = model.cpu()
              image_cpu = image.cpu()
              
              with torch.no_grad():
                  feature = model_cpu.visual(image_cpu)
              
              # Move result back to original device if needed
              if device.type == 'cuda':
                  feature = feature.cuda()
              
              return feature
          else:
              raise e
  
  def forward_new(self):
      def tmp(hidden_states, grid_thw=None):
          hidden_states = self.patch_embed(hidden_states)
          if grid_thw is not None:
              rotary_pos_emb = self.rot_pos_emb(grid_thw)
              cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                  dim=0, dtype=torch.int32
              )
              cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
              # Save tensors (move to CPU for numpy operations)
              np.save("./rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
              np.save("./cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())
          else:
              rotary_pos_emb = torch.from_numpy(np.load("./rotary_pos_emb.npy")).to(dtype=hidden_states.dtype, device=hidden_states.device)
              cu_seqlens = torch.from_numpy(np.load("./cu_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
          
          # Process blocks with memory management
          for i, blk in enumerate(self.blocks):
              hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
              
              # Clear cache every few blocks if memory aggressive
              if memory_aggressive and i % 4 == 0 and device.type == 'cuda':
                  torch.cuda.empty_cache()

          return self.merger(hidden_states)
      return tmp
  
  return export_onnx

def main():
  try:
      # Get user preferences
      device_choice, precision_choice, memory_aggressive = get_user_preferences()
      
      # Setup device and precision
      device, model_dtype, tensor_dtype = setup_device_and_precision(device_choice, precision_choice, memory_aggressive)
      
      # Initialize quantizer
      quantize = MakeInputEmbeds()
      
      # Load model with memory optimization
      model, tokenizer = load_model_with_memory_optimization(quantize, device, model_dtype, memory_aggressive)
      
      # Model configuration
      grid_t = 1
      grid_h = 28
      grid_w = 28
      merge_size = 2
      channel = 3
      temporal_patch_size = 2
      patch_size = 14
      
      # Create input tensor with smaller size if memory constrained
      if memory_aggressive:
          # Use smaller input size for testing
          pixel_values = torch.randn(1, 3, 224, 224, device=device, dtype=tensor_dtype)
          print("Using smaller input size (224x224) for memory optimization")
      else:
          pixel_values = torch.randn(1, 3, 392, 392, device=device, dtype=tensor_dtype)
      
      # Setup processing function
      export_onnx = process_with_checkpointing(model, pixel_values, device, memory_aggressive)
      model.forward = export_onnx
      
      # ONNX Export with memory management
      print(f"\n🔄 Exporting to ONNX...")
      os.makedirs("onnx", exist_ok=True)
      onnx_path = f"./onnx/{quantize.model_name}.onnx"
      
      try:
          # Clear memory before export
          clear_memory()
          
          # Ensure float32 for ONNX
          if model_dtype != torch.float32 or tensor_dtype != torch.float32:
              print("Converting to float32 for ONNX compatibility...")
              if device.type == 'cuda' and not memory_aggressive:
                  export_model = model.to(torch.float32)
                  export_input = pixel_values.to(torch.float32)
              else:
                  # Use CPU for conversion if memory constrained
                  export_model = model.cpu().to(torch.float32)
                  export_input = pixel_values.cpu().to(torch.float32)
          else:
              export_model = model
              export_input = pixel_values
          
          torch.onnx.export(export_model, export_input, onnx_path, opset_version=18)
          print(f"✓ ONNX export completed: {onnx_path}")
          
          # Clean up
          if export_model is not model:
              del export_model, export_input
          clear_memory()
      
      except RuntimeError as e:
          if "out of memory" in str(e).lower():
              print(f"❌ GPU OOM during ONNX export. Using CPU...")
              clear_memory()
              
              # Force CPU export
              model_cpu = model.cpu().to(torch.float32)
              input_cpu = pixel_values.cpu().to(torch.float32)
              torch.onnx.export(model_cpu, input_cpu, onnx_path, opset_version=18)
              print(f"✓ ONNX export completed with CPU")
              del model_cpu, input_cpu
              clear_memory()
          else:
              raise e
      
      # Continue with RKNN and RKLLM as before...
      print(f"\n🔧 Building RKNN model...")
      target_platform = "rk3588"
      
      rknn = RKNN(verbose=False)
      rknn.config(target_platform=target_platform, 
                 mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], 
                 std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]])
      rknn.load_onnx(onnx_path)
      rknn.build(do_quantization=False, dataset=None)
      
      os.makedirs("rknn", exist_ok=True)
      savepath = f'./rknn/{quantize.model_name}'
      rknn.export_rknn(f'{savepath}.rknn')
      print(f"✓ RKNN model saved: {savepath}.rknn")
      
      # RKLLM Build
      print(f"\n⚙️ Building RKLLM model...")
      llm = RKLLM()
      
      ret = llm.load_huggingface(model=quantize.path, device='cpu')
      if ret != 0:
          print('❌ Load model failed!')
          return ret
      
      dataset = 'data/inputs.json'
      qparams = None
      ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8',
                      quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3, 
                      extra_qparams=qparams, dataset=dataset)
      
      if ret != 0:
          print('❌ Build model failed!')
          return ret
      
      ret = llm.export_rkllm(f'{savepath}.rkllm')
      if ret != 0:
          print('❌ Export model failed!')
          return ret
      
      print(f"✓ RKLLM model saved: {savepath}.rkllm")
      
      # Final summary
      clear_memory()
      if device.type == 'cuda':
          total_mem, allocated, reserved, free = get_gpu_memory_info()
          print(f"\n📊 Final GPU memory: {allocated:.1f}GB used, {free:.1f}GB free")
      
      print(f"\n🎉 Quantization completed successfully!")
      print(f"📁 Output files:")
      print(f"   - ONNX: {onnx_path}")
      print(f"   - RKNN: {savepath}.rknn")
      print(f"   - RKLLM: {savepath}.rkllm")

  except RuntimeError as e:
      if "out of memory" in str(e).lower():
          print(f"\n❌ CUDA Out of Memory Error!")
          print(f"💡 Suggestions:")
          print(f"   1. Use CPU mode (option 2)")
          print(f"   2. Use aggressive memory optimization (option 3)")
          print(f"   3. Close other GPU applications")
          print(f"   4. Restart the script and try Float16 precision")
      else:
          print(f"\n❌ Runtime error: {e}")
      sys.exit(1)
  except KeyboardInterrupt:
      print("\n❌ Process interrupted by user")
      clear_memory()
      sys.exit(1)
  except Exception as e:
      print(f"\n❌ Unexpected error: {e}")
      clear_memory()
      sys.exit(1)

if __name__ == "__main__":
  main()
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

def get_user_preferences():
  """Get user preferences for device and precision"""
  print("=== Model Conversion Configuration ===")
  
  # Device selection
  cuda_available = torch.cuda.is_available()
  if cuda_available:
      print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
      print("Device options:")
      print("1. GPU (CUDA) - Faster processing")
      print("2. CPU - More compatible, slower")
      
      while True:
          choice = input("Choose device (1 for GPU, 2 for CPU): ").strip()
          if choice == '1':
              device_choice = 'gpu'
              break
          elif choice == '2':
              device_choice = 'cpu'
              break
          else:
              print("Please enter 1 or 2")
  else:
      print("CUDA is not available. Using CPU.")
      device_choice = 'cpu'
  
  # Precision selection (only for GPU)
  if device_choice == 'gpu':
      print("\nPrecision options:")
      print("1. Float32 - Higher precision, more memory usage")
      print("2. Float16 - Faster processing, less memory, potential precision loss")
      print("3. Mixed Precision - Balance of speed and accuracy")
      
      while True:
          choice = input("Choose precision (1/2/3): ").strip()
          if choice == '1':
              precision_choice = 'float32'
              break
          elif choice == '2':
              precision_choice = 'float16'
              break
          elif choice == '3':
              precision_choice = 'mixed'
              break
          else:
              print("Please enter 1, 2, or 3")
  else:
      precision_choice = 'float32'  # CPU always uses float32
  
  # Memory optimization
  print("\nMemory optimization:")
  print("1. Standard - Normal memory usage")
  print("2. Optimized - Lower memory usage, might be slower")
  
  while True:
      choice = input("Choose memory mode (1/2): ").strip()
      if choice == '1':
          memory_opt = False
          break
      elif choice == '2':
          memory_opt = True
          break
      else:
          print("Please enter 1 or 2")
  
  return device_choice, precision_choice, memory_opt

def setup_device_and_precision(device_choice, precision_choice):
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
  elif precision_choice == 'mixed':
      model_dtype = torch.float32
      tensor_dtype = torch.float32
      print(f"✓ Using Mixed precision (Float32 model with AMP)")
  else:
      model_dtype = torch.float32
      tensor_dtype = torch.float32
      print(f"✓ Using Float32 precision")
  
  return device, model_dtype, tensor_dtype

def load_model_with_config(quantize, device, model_dtype, memory_opt):
  """Load model with specified configuration"""
  print(f"\n📥 Loading model...")
  
  load_kwargs = {
      'torch_dtype': model_dtype,
      'trust_remote_code': True
  }
  
  if memory_opt:
      load_kwargs['low_cpu_mem_usage'] = True
      if device.type == 'cuda':
          load_kwargs['device_map'] = "auto"
  
  model = Qwen2VLForConditionalGeneration.from_pretrained(
      quantize.path, **load_kwargs
  ).eval()
  
  tokenizer = AutoTokenizer.from_pretrained(
      quantize.path, trust_remote_code=True, use_fast=False
  )
  
  # Move model to device if not using device_map
  if not memory_opt or device.type == 'cpu':
      model = model.to(device)
  
  print(f"✓ Model loaded successfully")
  return model, tokenizer

def print_memory_usage(device):
  """Print current memory usage"""
  if device.type == 'cuda':
      allocated = torch.cuda.memory_allocated() / 1024**3
      cached = torch.cuda.memory_reserved() / 1024**3
      print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def main():
  # Get user preferences
  device_choice, precision_choice, memory_opt = get_user_preferences()
  
  # Setup device and precision
  device, model_dtype, tensor_dtype = setup_device_and_precision(device_choice, precision_choice)
  
  # Initialize quantizer
  quantize = MakeInputEmbeds()
  
  # Load model
  model, tokenizer = load_model_with_config(quantize, device, model_dtype, memory_opt)
  
  # Print initial memory usage
  print_memory_usage(device)
  
  # Model configuration
  grid_t = 1
  grid_h = 28
  grid_w = 28
  merge_size = 2
  channel = 3
  temporal_patch_size = 2
  patch_size = 14
  
  def export_onnx(image):
      # Ensure image is on the correct device
      image = image.to(device)
      
      patches = image.repeat(temporal_patch_size, 1, 1, 1)
      patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
      patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
      flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
      
      model.visual.forward = forward_new(model.visual)
      
      # Apply precision strategy
      if precision_choice == 'mixed' and device.type == 'cuda':
          with torch.cuda.amp.autocast():
              with torch.no_grad():
                  feature = model.visual(flatten_patches)
          # Ensure output is float32 for downstream processing
          return feature.float() if feature.dtype != torch.float32 else feature
      else:
          with torch.no_grad():
              feature = model.visual(flatten_patches)
          return feature
  
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
          
          for blk in self.blocks:
              hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

          return self.merger(hidden_states)
      return tmp
  
  # Create input tensor
  pixel_values = torch.randn(1, 3, 392, 392, device=device, dtype=tensor_dtype)
  
  model.forward = export_onnx
  
  # ONNX Export
  print(f"\n🔄 Exporting to ONNX...")
  os.makedirs("onnx", exist_ok=True)
  onnx_path = f"./onnx/{quantize.model_name}.onnx"
  
  try:
      # For ONNX export, ensure float32 precision
      if model_dtype != torch.float32 or tensor_dtype != torch.float32:
          print("Converting to float32 for ONNX compatibility...")
          export_model = model.to(torch.float32)
          export_input = pixel_values.to(torch.float32)
      else:
          export_model = model
          export_input = pixel_values
      
      torch.onnx.export(export_model, export_input, onnx_path, opset_version=18)
      print(f"✓ ONNX export completed: {onnx_path}")
      
      # Clean up if we created copies
      if export_model is not model:
          del export_model, export_input
          if device.type == 'cuda':
              torch.cuda.empty_cache()
  
  except Exception as e:
      print(f"❌ ONNX export failed: {e}")
      if device.type == 'cuda':
          print("Trying CPU fallback...")
          try:
              model_cpu = model.cpu().to(torch.float32)
              input_cpu = pixel_values.cpu().to(torch.float32)
              torch.onnx.export(model_cpu, input_cpu, onnx_path, opset_version=18)
              print(f"✓ ONNX export completed with CPU fallback")
              del model_cpu, input_cpu
              torch.cuda.empty_cache()
          except Exception as e2:
              print(f"❌ CPU fallback also failed: {e2}")
              return
  
  print_memory_usage(device)
  
  # RKNN Build
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
  
  # Export rkllm model
  ret = llm.export_rkllm(f'{savepath}.rkllm')
  if ret != 0:
      print('❌ Export model failed!')
      return ret
  
  print(f"✓ RKLLM model saved: {savepath}.rkllm")
  
  # Final cleanup and summary
  if device.type == 'cuda':
      torch.cuda.empty_cache()
      max_memory = torch.cuda.max_memory_allocated() / 1024**3
      print(f"\n📊 Peak GPU memory usage: {max_memory:.2f}GB")
  
  print(f"\n🎉 Quantization completed successfully!")
  print(f"📁 Output files:")
  print(f"   - ONNX: {onnx_path}")
  print(f"   - RKNN: {savepath}.rknn")
  print(f"   - RKLLM: {savepath}.rkllm")

if __name__ == "__main__":
  try:
      main()
  except KeyboardInterrupt:
      print("\n❌ Process interrupted by user")
      sys.exit(1)
  except Exception as e:
      print(f"\n❌ Unexpected error: {e}")
      sys.exit(1)
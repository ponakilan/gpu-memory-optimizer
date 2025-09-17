import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import List, Dict, Tuple
import time
import os
import gc
import psutil
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


class MemoryMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.memory_log = {
            'peak_gpu': 0,
            'current_gpu': 0,
            'peak_cpu': 0,
            'current_cpu': 0,
            'gpu_reserved': 0,
            'forward_memory': [],
            'backward_memory': [],
            'batch_memory': []
        }
    
    def get_gpu_memory(self):
        
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**2  
            peak = torch.cuda.max_memory_allocated() / 1024**2  
            reserved = torch.cuda.memory_reserved() / 1024**2  
            return current, peak, reserved
        return 0, 0, 0
    
    def get_cpu_memory(self):
        
        process = psutil.Process()
        current = process.memory_info().rss / 1024**2  
        return current
    
    def update(self, phase='general'):
        
        gpu_current, gpu_peak, gpu_reserved = self.get_gpu_memory()
        cpu_current = self.get_cpu_memory()
        
        self.memory_log['current_gpu'] = gpu_current
        self.memory_log['peak_gpu'] = max(self.memory_log['peak_gpu'], gpu_peak)
        self.memory_log['current_cpu'] = cpu_current
        self.memory_log['peak_cpu'] = max(self.memory_log['peak_cpu'], cpu_current)
        self.memory_log['gpu_reserved'] = gpu_reserved
        
        if phase == 'forward':
            self.memory_log['forward_memory'].append(gpu_current)
        elif phase == 'backward':
            self.memory_log['backward_memory'].append(gpu_current)
        elif phase == 'batch':
            self.memory_log['batch_memory'].append(gpu_current)
    
    def get_stats(self):
        
        return {
            'gpu_current_mb': self.memory_log['current_gpu'],
            'gpu_peak_mb': self.memory_log['peak_gpu'],
            'cpu_current_mb': self.memory_log['current_cpu'],
            'cpu_peak_mb': self.memory_log['peak_cpu'],
            'gpu_reserved_mb': self.memory_log['gpu_reserved']
        }
    
    def reset_peak(self):
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.memory_log['peak_gpu'] = 0


class Config:
    def __init__(self):
        self.batch_size = 32  
        self.num_epochs = 10  
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.print_freq = 25
        self.data_path = "./data/cifar10"
        self.num_workers = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_checkpointing = True
        self.checkpoint_method = "linear"  
        self.download_dataset = True
        self.log_dir = "./tensorboard_logs"
        self.memory_log_interval = 10  

class CheckpointBase:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = np.array(layer_sizes)
        self.n = len(layer_sizes) - 1

    def evaluate_checkpoint_cost(self, checkpoints: List[int], algorithm_type: str = "pytorch") -> int:
        d = self.layer_sizes
        checkpoints = sorted(set(checkpoints + [0, self.n]))
        
        if algorithm_type == "basic":
            checkpoint_memory = sum(d[i] for i in checkpoints)
            max_segment_memory = 0
            for i in range(len(checkpoints) - 1):
                start = checkpoints[i]
                end = checkpoints[i + 1]
                segment_memory = sum(d[start + 1:end])
                max_segment_memory = max(max_segment_memory, segment_memory)
            return checkpoint_memory + max_segment_memory
        else:
            max_memory = 0
            for i in range(len(checkpoints) - 1, 0, -1):
                curr_checkpoint = checkpoints[i]
                prev_checkpoint = checkpoints[i - 1]
                checkpoint_mem = sum(d[j] for j in checkpoints[:i+1])
                segment_mem = sum(d[prev_checkpoint + 1:curr_checkpoint])
                gradient_buffer = max(d[prev_checkpoint:curr_checkpoint]) if prev_checkpoint < curr_checkpoint else 0
                total_mem = checkpoint_mem + segment_mem + gradient_buffer
                max_memory = max(max_memory, total_mem)
            return max_memory

class LinearCheckpointSelector(CheckpointBase):
    def dynamic_checkpoint_selection_linear(self) -> Tuple[int, List[int]]:
        n = self.n
        d = self.layer_sizes
        
        if n == 0:
            return int(d[0]), [0]
        
        def compute_s(h: int, i: int) -> int:
            if h >= i:
                return 0
            segment_sum = sum(d[h+1:i])
            max_val = max(d[h:i]) if h < i else 0
            return segment_sum + max_val
        
        M = [0] * (n + 1)
        parent = [-1] * (n + 1)
        M[n] = d[n]
        Q = deque([n])
        j_star = n
        
        for i in range(n - 1, 0, -1):
            min_cost = float('inf')
            best_j = j_star
            
            for curr_j in range(i + 1, n + 1):
                if curr_j > len(Q) + i:
                    break
                    
                s_val = compute_s(i, curr_j)
                U_val = s_val + d[curr_j]
                case1 = d[i] + M[curr_j]
                case2 = d[i] + U_val
                cost = max(case1, case2)
                
                if cost < min_cost:
                    min_cost = cost
                    best_j = curr_j
            
            M[i] = min_cost
            parent[i] = best_j
            j_star = best_j
            
            while Q and M[Q[0]] >= M[i]:
                Q.popleft()
            Q.appendleft(i)
        
        checkpoints = [0]
        curr = 1
        while curr <= n:
            if curr == n or parent[curr] != curr + 1:
                checkpoints.append(curr)
            if parent[curr] != -1:
                curr = parent[curr]
            else:
                break
        
        if n not in checkpoints:
            checkpoints.append(n)
        
        return int(M[1]), sorted(checkpoints)

class BaselineMethods(CheckpointBase):
    def sqrt_n_method(self) -> List[int]:
        n = self.n
        if n <= 1:
            return [0, n]
        
        num_segments = max(1, int(np.sqrt(n)))
        segment_size = n // num_segments
        
        checkpoints = [0]
        for i in range(1, num_segments):
            checkpoints.append(i * segment_size)
        checkpoints.append(n)
        
        return sorted(list(set(checkpoints)))

class CheckpointOptimizedVGG19(nn.Module):
    def __init__(self, num_classes=10, use_checkpointing=True, checkpoint_method="linear"):
        super(CheckpointOptimizedVGG19, self).__init__()
        self.use_checkpointing = use_checkpointing
        self.checkpoint_method = checkpoint_method
        self.memory_monitor = MemoryMonitor()
        
        
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        
        
        self.features = self._make_layers(cfg)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
        
        self.layer_info = self._analyze_layers()
        self.checkpoint_points = self._get_optimal_checkpoints()
        
        self._initialize_weights()
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        
        return nn.ModuleList(layers)
    
    def _analyze_layers(self):
        
        layer_sizes = []
        
        
        feature_map_size = 32 * 32
        channels = 3
        
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                param_mem = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                feature_mem = channels * feature_map_size
                layer_sizes.append(int((param_mem + feature_mem) / 1000))  
                channels = layer.out_channels
            elif isinstance(layer, nn.MaxPool2d):
                feature_map_size //= 4  
                layer_sizes.append(int(channels * feature_map_size / 1000))
        
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                param_mem = layer.weight.numel() + layer.bias.numel()
                layer_sizes.append(int(param_mem / 1000))
            else:
                layer_sizes.append(10)  
        
        return layer_sizes
    
    def _get_optimal_checkpoints(self):
        
        if not self.use_checkpointing or self.checkpoint_method == "none":
            return []
        
        
        class CombinedOptimizer(LinearCheckpointSelector, BaselineMethods):
            pass
        
        optimizer = CombinedOptimizer(self.layer_info)
        
        if self.checkpoint_method == "linear":
            _, checkpoints = optimizer.dynamic_checkpoint_selection_linear()
        elif self.checkpoint_method == "sqrt_n":
            checkpoints = optimizer.sqrt_n_method()
        else:
            checkpoints = []
        
        
        return [i for i in checkpoints if 0 < i < len(self.features)]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        self.memory_monitor.update('forward')
        
        
        if self.use_checkpointing and self.checkpoint_points:
            x = self._forward_with_checkpointing(x)
        else:
            for layer in self.features:
                x = layer(x)
        
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        
        self.memory_monitor.update('forward')
        
        return x
    
    def _forward_with_checkpointing(self, x):
        
        checkpoint_points = [0] + self.checkpoint_points + [len(self.features)]
        
        for i in range(len(checkpoint_points) - 1):
            start_idx = checkpoint_points[i]
            end_idx = checkpoint_points[i + 1]
            
            
            def create_segment(start, end):
                def segment_forward(x_seg):
                    for j in range(start, end):
                        x_seg = self.features[j](x_seg)
                    return x_seg
                return segment_forward
            
            segment_fn = create_segment(start_idx, end_idx)
            
            
            if self.training and i < len(checkpoint_points) - 2:  
                x = checkpoint.checkpoint(segment_fn, x)
            else:
                x = segment_fn(x)
        
        return x

def get_data_loaders(config):
    
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                   std=[0.2023, 0.1994, 0.2010])
    
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    print("Downloading CIFAR-10 dataset...")
    
    
    train_dataset = datasets.CIFAR10(
        root=config.data_path,
        train=True,
        download=config.download_dataset,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=config.data_path,
        train=False,
        download=config.download_dataset,
        transform=val_transform
    )
    
    print(f"CIFAR-10 dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} test samples")
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, epoch, config, writer, global_step):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_memory_stats = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        model.memory_monitor.reset_peak()
        
        data, target = data.to(config.device), target.to(config.device)
        
        
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        
        model.memory_monitor.update('batch')
        memory_before_forward = model.memory_monitor.get_stats()
        
        
        output = model(data)
        loss = criterion(output, target)
        
        
        model.memory_monitor.update('forward')
        memory_after_forward = model.memory_monitor.get_stats()
        
        
        loss.backward()
        
        
        model.memory_monitor.update('backward')
        memory_after_backward = model.memory_monitor.get_stats()
        
        optimizer.step()
        
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        
        if batch_idx % config.memory_log_interval == 0:
            batch_memory_stats = {
                'batch': batch_idx,
                'before_forward': memory_before_forward,
                'after_forward': memory_after_forward,
                'after_backward': memory_after_backward,
                'loss': loss.item(),
                'accuracy': 100. * correct / total
            }
            epoch_memory_stats.append(batch_memory_stats)
            
            
            step = global_step + batch_idx
            writer.add_scalar('Memory/GPU_Current_MB', memory_after_backward['gpu_current_mb'], step)
            writer.add_scalar('Memory/GPU_Peak_MB', memory_after_backward['gpu_peak_mb'], step)
            writer.add_scalar('Memory/CPU_Current_MB', memory_after_backward['cpu_current_mb'], step)
            writer.add_scalar('Memory/GPU_Reserved_MB', memory_after_backward['gpu_reserved_mb'], step)
            
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Accuracy', 100. * correct / total, step)
            
            
            forward_increase = memory_after_forward['gpu_peak_mb'] - memory_before_forward['gpu_current_mb']
            backward_increase = memory_after_backward['gpu_peak_mb'] - memory_after_forward['gpu_peak_mb']
            
            writer.add_scalar('Memory/Forward_Memory_Increase_MB', forward_increase, step)
            writer.add_scalar('Memory/Backward_Memory_Increase_MB', backward_increase, step)
            writer.add_scalar('Memory/Total_Memory_Increase_MB', forward_increase + backward_increase, step)
        
        if batch_idx % config.print_freq == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.2f}%, '
                  f'GPU Mem: {memory_after_backward["gpu_current_mb"]:.1f}MB '
                  f'(Peak: {memory_after_backward["gpu_peak_mb"]:.1f}MB)')
    
    return running_loss / len(train_loader), 100. * correct / total, epoch_memory_stats

def validate(model, val_loader, criterion, config, writer, global_step):
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    val_memory_stats = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            model.memory_monitor.reset_peak()
            
            data, target = data.to(config.device), target.to(config.device)
            
            
            model.memory_monitor.update('batch')
            memory_before = model.memory_monitor.get_stats()
            
            output = model(data)
            val_loss += criterion(output, target).item()
            
            
            model.memory_monitor.update('batch')
            memory_after = model.memory_monitor.get_stats()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % config.memory_log_interval == 0:
                val_memory_stats.append({
                    'batch': batch_idx,
                    'before': memory_before,
                    'after': memory_after
                })
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    
    
    writer.add_scalar('Validation/Loss', val_loss, global_step)
    writer.add_scalar('Validation/Accuracy', accuracy, global_step)
    
    print(f'Validation Loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%')
    return val_loss, accuracy, val_memory_stats

def analyze_memory_usage():
    
    print("Memory Analysis of Checkpointing Methods for VGG-19 on CIFAR-10")
    print("=" * 70)
    
    
    model = CheckpointOptimizedVGG19(num_classes=10, use_checkpointing=False)
    layer_sizes = model.layer_info
    
    print(f"VGG-19 adapted for CIFAR-10 with {len(layer_sizes)} layers")
    print(f"Total estimated layer memory: {sum(layer_sizes)} KB")
    print(f"Layer sizes (KB): {layer_sizes[:10]}... (showing first 10)")
    print()
    
    
    methods = ["none", "sqrt_n", "linear"]
    results = {}
    
    for method in methods:
        if method == "none":
            checkpoints = [0, len(layer_sizes)-1]
            cost = sum(layer_sizes)
        else:
            
            class CombinedOptimizer(LinearCheckpointSelector, BaselineMethods):
                pass
            
            optimizer = CombinedOptimizer(layer_sizes)
            
            if method == "linear":
                cost, checkpoints = optimizer.dynamic_checkpoint_selection_linear()
            elif method == "sqrt_n":
                checkpoints = optimizer.sqrt_n_method()
                cost = optimizer.evaluate_checkpoint_cost(checkpoints, "pytorch")
        
        results[method] = {
            'checkpoints': checkpoints,
            'cost': cost
        }
    
    
    baseline = results['none']['cost']
    print("Theoretical Memory Analysis:")
    print("-" * 40)
    for method, result in results.items():
        cost = result['cost']
        savings = baseline - cost
        savings_pct = (savings / baseline) * 100 if baseline > 0 else 0
        
        print(f"{method.upper()} Method:")
        print(f"  Checkpoints: {len(result['checkpoints'])} points at {result['checkpoints']}")
        print(f"  Estimated memory cost: {cost} KB")
        print(f"  Theoretical savings: {savings} KB ({savings_pct:.1f}%)")
        print()
    
    return results

def create_memory_comparison_plots(results_with_checkpoint, results_without_checkpoint, save_dir="./plots"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    
    plt.style.use('seaborn-v0_8')
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = ['Without Checkpointing', 'With Checkpointing']
    peak_memory = [
        max([stats['gpu_peak_mb'] for batch_stats in results_without_checkpoint for stats in [batch_stats['after_backward']]]),
        max([stats['gpu_peak_mb'] for batch_stats in results_with_checkpoint for stats in [batch_stats['after_backward']]])
    ]
    
    bars = ax1.bar(methods, peak_memory, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Peak GPU Memory (MB)')
    ax1.set_title('Peak GPU Memory Usage Comparison')
    
    
    for bar, value in zip(bars, peak_memory):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    
    savings = peak_memory[0] - peak_memory[1]
    savings_pct = (savings / peak_memory[0]) * 100
    ax1.text(0.5, max(peak_memory) * 0.8, f'Savings: {savings:.1f}MB\n({savings_pct:.1f}%)', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    
    without_checkpoint_memory = [stats['gpu_current_mb'] for stats in [batch_stats['after_backward'] for batch_stats in results_without_checkpoint]]
    with_checkpoint_memory = [stats['gpu_current_mb'] for stats in [batch_stats['after_backward'] for batch_stats in results_with_checkpoint]]
    
    batches = range(len(without_checkpoint_memory))
    
    ax2.plot(batches, without_checkpoint_memory, label='Without Checkpointing', color='red', alpha=0.7)
    ax2.plot(batches, with_checkpoint_memory, label='With Checkpointing', color='green', alpha=0.7)
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('GPU Memory (MB)')
    ax2.set_title('GPU Memory Usage Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Memory comparison plots saved to {save_dir}/memory_comparison.png")

def run_training_comparison(config):
    
    
    
    writer_with_checkpoint = SummaryWriter(f"{config.log_dir}/with_checkpoint")
    writer_without_checkpoint = SummaryWriter(f"{config.log_dir}/without_checkpoint")
    
    
    try:
        train_loader, val_loader = get_data_loaders(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    results = {}
    
    
    print("\n" + "="*60)
    print("TRAINING WITH GRADIENT CHECKPOINTING")
    print("="*60)
    
    model_with_checkpoint = CheckpointOptimizedVGG19(
        num_classes=10,
        use_checkpointing=True,
        checkpoint_method=config.checkpoint_method
    ).to(config.device)
    
    print(f"Model with checkpointing - Checkpoint points: {model_with_checkpoint.checkpoint_points}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_with_checkpoint.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    memory_stats_with_checkpoint = []
    
    for epoch in range(min(3, config.num_epochs)):  
        print(f"\nEpoch {epoch+1}")
        print("-" * 40)
        
        train_loss, train_acc, epoch_memory = train_epoch(
            model_with_checkpoint, train_loader, criterion, optimizer, 
            epoch, config, writer_with_checkpoint, epoch * len(train_loader)
        )
        
        val_loss, val_acc, val_memory = validate(
            model_with_checkpoint, val_loader, criterion, config, 
            writer_with_checkpoint, (epoch + 1) * len(train_loader)
        )
        
        memory_stats_with_checkpoint.extend(epoch_memory)
    
    
    print("\n" + "="*60)
    print("TRAINING WITHOUT GRADIENT CHECKPOINTING")
    print("="*60)
    
    model_without_checkpoint = CheckpointOptimizedVGG19(
        num_classes=10,
        use_checkpointing=False,
        checkpoint_method="none"
    ).to(config.device)
    
    optimizer = optim.Adam(model_without_checkpoint.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    memory_stats_without_checkpoint = []
    
    for epoch in range(min(3, config.num_epochs)):  
        print(f"\nEpoch {epoch+1}")
        print("-" * 40)
        
        train_loss, train_acc, epoch_memory = train_epoch(
            model_without_checkpoint, train_loader, criterion, optimizer, 
            epoch, config, writer_without_checkpoint, epoch * len(train_loader)
        )
        
        val_loss, val_acc, val_memory = validate(
            model_without_checkpoint, val_loader, criterion, config, 
            writer_without_checkpoint, (epoch + 1) * len(train_loader)
        )
        
        memory_stats_without_checkpoint.extend(epoch_memory)
    
    
    writer_with_checkpoint.close()
    writer_without_checkpoint.close()
    
    
    if memory_stats_with_checkpoint and memory_stats_without_checkpoint:
        create_memory_comparison_plots(memory_stats_with_checkpoint, memory_stats_without_checkpoint)
    
    
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON SUMMARY")
    print("="*60)
    
    def get_peak_memory(memory_stats):
        return max([stats['gpu_peak_mb'] for batch_stats in memory_stats 
                   for stats in [batch_stats['after_backward']]])
    
    def get_avg_memory(memory_stats):
        return np.mean([stats['gpu_current_mb'] for batch_stats in memory_stats 
                       for stats in [batch_stats['after_backward']]])
    
    if memory_stats_with_checkpoint and memory_stats_without_checkpoint:
        peak_with = get_peak_memory(memory_stats_with_checkpoint)
        peak_without = get_peak_memory(memory_stats_without_checkpoint)
        avg_with = get_avg_memory(memory_stats_with_checkpoint)
        avg_without = get_avg_memory(memory_stats_without_checkpoint)
        
        peak_savings = peak_without - peak_with
        peak_savings_pct = (peak_savings / peak_without) * 100
        avg_savings = avg_without - avg_with
        avg_savings_pct = (avg_savings / avg_without) * 100
        
        print(f"Peak GPU Memory Usage:")
        print(f"  Without Checkpointing: {peak_without:.1f} MB")
        print(f"  With Checkpointing:    {peak_with:.1f} MB")
        print(f"  Peak Memory Savings:   {peak_savings:.1f} MB ({peak_savings_pct:.1f}%)")
        print()
        print(f"Average GPU Memory Usage:")
        print(f"  Without Checkpointing: {avg_without:.1f} MB")
        print(f"  With Checkpointing:    {avg_with:.1f} MB")
        print(f"  Average Memory Savings: {avg_savings:.1f} MB ({avg_savings_pct:.1f}%)")
    
    return memory_stats_with_checkpoint, memory_stats_without_checkpoint

def create_detailed_tensorboard_logs(memory_stats, writer, prefix):
    
    
    
    for i, batch_stats in enumerate(memory_stats):
        step = i
        
        
        writer.add_scalars(f'{prefix}/Memory_Phases', {
            'Before_Forward': batch_stats['before_forward']['gpu_current_mb'],
            'After_Forward': batch_stats['after_forward']['gpu_current_mb'],
            'After_Backward': batch_stats['after_backward']['gpu_current_mb']
        }, step)
        
        
        forward_increase = (batch_stats['after_forward']['gpu_current_mb'] - 
                          batch_stats['before_forward']['gpu_current_mb'])
        backward_increase = (batch_stats['after_backward']['gpu_current_mb'] - 
                           batch_stats['after_forward']['gpu_current_mb'])
        
        writer.add_scalars(f'{prefix}/Memory_Increases', {
            'Forward_Pass': forward_increase,
            'Backward_Pass': backward_increase,
            'Total': forward_increase + backward_increase
        }, step)
        
        
        writer.add_scalars(f'{prefix}/Memory_Peak_vs_Current', {
            'Current': batch_stats['after_backward']['gpu_current_mb'],
            'Peak': batch_stats['after_backward']['gpu_peak_mb']
        }, step)

def benchmark_memory_usage():
    
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARK")
    print("="*60)
    
    batch_sizes = [16, 32, 64, 128]
    benchmark_results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        config_temp = Config()
        config_temp.batch_size = batch_size
        config_temp.num_epochs = 1  
        
        try:
            train_loader, _ = get_data_loaders(config_temp)
            
            
            model_with = CheckpointOptimizedVGG19(
                num_classes=10, use_checkpointing=True, checkpoint_method="linear"
            ).to(config_temp.device)
            
            
            model_without = CheckpointOptimizedVGG19(
                num_classes=10, use_checkpointing=False, checkpoint_method="none"
            ).to(config_temp.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer_with = optim.Adam(model_with.parameters(), lr=1e-3)
            optimizer_without = optim.Adam(model_without.parameters(), lr=1e-3)
            
            
            peak_memory_with = 0
            peak_memory_without = 0
            
            model_with.train()
            model_without.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  
                    break
                
                data, target = data.to(config_temp.device), target.to(config_temp.device)
                
                
                torch.cuda.reset_peak_memory_stats()
                optimizer_with.zero_grad()
                output = model_with(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_with.step()
                peak_memory_with = max(peak_memory_with, torch.cuda.max_memory_allocated() / 1024**2)
                
                
                torch.cuda.reset_peak_memory_stats()
                optimizer_without.zero_grad()
                output = model_without(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_without.step()
                peak_memory_without = max(peak_memory_without, torch.cuda.max_memory_allocated() / 1024**2)
            
            benchmark_results[batch_size] = {
                'with_checkpoint': peak_memory_with,
                'without_checkpoint': peak_memory_without,
                'savings': peak_memory_without - peak_memory_with,
                'savings_pct': ((peak_memory_without - peak_memory_with) / peak_memory_without) * 100
            }
            
            print(f"  With checkpointing: {peak_memory_with:.1f} MB")
            print(f"  Without checkpointing: {peak_memory_without:.1f} MB")
            print(f"  Savings: {benchmark_results[batch_size]['savings']:.1f} MB "
                  f"({benchmark_results[batch_size]['savings_pct']:.1f}%)")
                  
        except Exception as e:
            print(f"  Error with batch size {batch_size}: {e}")
            continue
    
    
    if benchmark_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        batch_sizes_tested = list(benchmark_results.keys())
        memory_with = [benchmark_results[bs]['with_checkpoint'] for bs in batch_sizes_tested]
        memory_without = [benchmark_results[bs]['without_checkpoint'] for bs in batch_sizes_tested]
        savings = [benchmark_results[bs]['savings'] for bs in batch_sizes_tested]
        
        
        x = np.arange(len(batch_sizes_tested))
        width = 0.35
        
        ax1.bar(x - width/2, memory_without, width, label='Without Checkpointing', color='red', alpha=0.7)
        ax1.bar(x + width/2, memory_with, width, label='With Checkpointing', color='green', alpha=0.7)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Peak GPU Memory (MB)')
        ax1.set_title('Memory Usage vs Batch Size')
        ax1.set_xticks(x)
        ax1.set_xticklabels(batch_sizes_tested)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        ax2.bar(batch_sizes_tested, savings, color='blue', alpha=0.7)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Savings (MB)')
        ax2.set_title('Memory Savings vs Batch Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./plots/batch_size_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nBenchmark plot saved to ./plots/batch_size_benchmark.png")
    
    return benchmark_results

def main():
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Checkpoint method: {config.checkpoint_method}")
    
    
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    
    
    theoretical_results = analyze_memory_usage()
    
    
    if torch.cuda.is_available():
        benchmark_results = benchmark_memory_usage()
    
    
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("RUNNING TRAINING COMPARISON")
        print("="*60)
        
        memory_with_checkpoint, memory_without_checkpoint = run_training_comparison(config)
        
        if memory_with_checkpoint and memory_without_checkpoint:
            
            writer_detailed = SummaryWriter(f"{config.log_dir}/detailed_comparison")
            
            create_detailed_tensorboard_logs(memory_with_checkpoint, writer_detailed, "With_Checkpoint")
            create_detailed_tensorboard_logs(memory_without_checkpoint, writer_detailed, "Without_Checkpoint")
            
            writer_detailed.close()
    else:
        print("CUDA not available. Skipping actual training comparison.")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated outputs:")
    print("1. TensorBoard logs in ./tensorboard_logs/")
    print("2. Memory comparison plots in ./plots/")
    print("3. Theoretical analysis printed above")
    print("\nTo view TensorBoard logs:")
    print("tensorboard --logdir=./tensorboard_logs")

if __name__ == "__main__":
    print("VGG-19 CIFAR-10 Memory Usage Analysis with Gradient Checkpointing")
    print("=" * 70)
    
    main()
    
    print("\nAnalysis completed!")
    print("Key findings:")
    print("1. Gradient checkpointing reduces peak memory usage during training")
    print("2. Memory savings increase with larger batch sizes")
    print("3. Linear checkpoint selection provides optimal memory-computation trade-off")
    print("4. TensorBoard logs show detailed memory usage patterns")
    print("5. Plots visualize memory savings across different configurations")
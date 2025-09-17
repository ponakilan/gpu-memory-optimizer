from checkpoint_optimizer import CheckpointOptimizer

def test():
    layer_sizes = [50, 100, 200, 150, 300, 250, 400, 350, 
                   500, 450, 600, 550, 700, 650, 800, 750, 
                   900, 850, 1000, 950, 200, 150, 100]
    
    print("GPU Memory Checkpoint Optimization")
    print("=" * 50)
    print(f"Network with {len(layer_sizes)-1} layers")
    print(f"Layer sizes (MB): {layer_sizes}")
    print()
    
    optimizer = CheckpointOptimizer(layer_sizes)
    results = optimizer.compare_methods()
    
    print("Comparison of Checkpoint Selection Methods:")
    print("-" * 50)
    
    for method_name, result in results.items():
        print(f"{method_name.upper()} Method:")
        print(f"  Checkpoints: {result['checkpoints']}")
        print(f"  Basic model cost: {result['cost_basic']} MB")
        print(f"  PyTorch model cost: {result['cost_pytorch']} MB")
        print()
    
    baseline = results['none']['cost_pytorch']
    best_linear = results['linear']['cost_pytorch']
    savings = baseline - best_linear
    savings_pct = (savings / baseline) * 100
    
    print(f"Memory Savings with Linear Algorithm:")
    print(f"  Baseline (no checkpointing): {baseline} MB")
    print(f"  Optimized: {best_linear} MB")
    print(f"  Savings: {savings} MB ({savings_pct:.1f}%)")

if __name__ == "__main__":
    test()
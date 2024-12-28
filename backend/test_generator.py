import time
from fine_tuning import FastCodeGenerator

def print_with_border(text):
    width = max(len(line) for line in text.split('\n')) + 4
    print('=' * width)
    for line in text.split('\n'):
        print(f"| {line:<{width-4}} |")
    print('=' * width)

def main():
    try:
        print_with_border("Initializing AI Models...\nThis may take a few minutes on first run.")
        
        # Initialize the generator
        generator = FastCodeGenerator()
        
        # Test planning stage
        print_with_border("Testing Planning Stage...")
        prompt = "Create a simple Python function to calculate fibonacci numbers"
        
        print("\nPrompt:", prompt)
        print("\nGenerating plan...")
        start = time.time()
        
        plan_result = generator.generate(prompt, stage="planning")
        
        print("\nPlan generated in {:.2f} seconds".format(time.time() - start))
        print("\nGenerated Plan:")
        print_with_border(plan_result['text'])
        
        # Test coding stage
        print_with_border("Testing Code Generation Stage...")
        code_prompt = f"# Based on this plan:\n{plan_result['text']}\n\n# Generate the code:"
        
        print("\nGenerating code...")
        start = time.time()
        
        code_result = generator.generate(code_prompt, stage="coding")
        
        print("\nCode generated in {:.2f} seconds".format(time.time() - start))
        print("\nGenerated Code:")
        print_with_border(code_result['text'])
        
        # Print performance metrics
        print_with_border(
            "Performance Metrics:\n"
            f"Planning Time: {plan_result['metrics']['planning_time']:.2f}s\n"
            f"Coding Time: {code_result['metrics']['coding_time']:.2f}s\n"
            f"Total Time: {code_result['metrics']['total_time']:.2f}s"
        )
        
        print_with_border("Test completed successfully!")
        
    except Exception as e:
        print_with_border(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

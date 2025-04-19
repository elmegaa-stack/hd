from .nf4 import *
import argparse
import time
import logging
import os
import sys

def main():
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Image generation using Hugging Face models")
    
    # Required arguments
    parser.add_argument("--prompts", nargs="+", required=True,
                      help="List of prompts to generate images from (enclose each in quotes)")
    
    # Optional arguments
    parser.add_argument("-m", "--model", choices=["dev", "full", "fast"],
                      default="dev", help="Model version to use")
    parser.add_argument("-s", "--seed", type=int, default=-1,
                      help="Seed for random generation")
    parser.add_argument("-r", "--resolution", choices=["1024x1024", "768x1360", "1360x768",
                      "880x1168", "1168x880", "1248x832", "832x1248"],
                      default="1024x1024", dest="res",
                      help="Output resolution")
    parser.add_argument("-o", "--output-dir", default="outputs",
                      help="Output directory for generated images")
    parser.add_argument("--prefix", default="output",
                      help="Filename prefix for generated images")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    print(f"⏳ Loading {args.model} model...")
    start_load = time.time()
    pipe, _ = load_models(args.model)
    print(f"✅ Model loaded in {time.time() - start_load:.2f}s")

    # Process all prompts
    resolution = tuple(map(int, args.res.split("x")))
    
    for idx, prompt in enumerate(args.prompts, start=1):
        print(f"\n🏷  Processing prompt {idx}/{len(args.prompts)}")
        print(f"📝 Prompt: {prompt[:80]}...")
        
        try:
            start_gen = time.time()
            image, seed = generate_image(
                pipe=pipe,
                model_type=args.model,
                prompt=prompt,
                resolution=resolution,
                seed=args.seed
            )
            
            filename = f"{args.prefix}_{idx}_{seed}.png"
            output_path = os.path.join(args.output_dir, filename)
            image.save(output_path)
            
            print(f"🖼  Saved to {output_path}")
            print(f"⏱  Generation time: {time.time() - start_gen:.2f}s")
            print(f"🌱 Seed: {seed}")

        except Exception as e:
            print(f"❌ Error processing prompt {idx}: {str(e)}")
            continue

if __name__ == "__main__":
    main()

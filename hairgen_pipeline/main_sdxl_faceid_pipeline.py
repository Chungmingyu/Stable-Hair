import argparse
from modules.hairstyle_generator_sdxl import HairstyleGeneratorSDXL

def main():
    parser = argparse.ArgumentParser(description="SDXL 참조 이미지 기반 헤어스타일 생성")
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output", default="hairstyle_results")
    parser.add_argument("--align_face", action="store_true", help="입력 이미지 얼굴 정렬 여부 (기본: False)")
    parser.add_argument("--samples", type=int, default=1, help="샘플 개수")
    parser.add_argument("--prompt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--s_scale", type=float, default=0.7)
    parser.add_argument("--no_shortcut", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    generator = HairstyleGeneratorSDXL()
    generator.generate_from_references(
        input_image_path=args.input,
        reference_dir=args.reference_dir,
        output_dir=args.output,
        align_face=args.align_face,
        num_per_reference=args.samples,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        s_scale=args.s_scale,
        shortcut=not args.no_shortcut,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 
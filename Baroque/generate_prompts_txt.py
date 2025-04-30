import os

baroque_dir = os.path.dirname(__file__)
prompt_files = [f for f in os.listdir(baroque_dir) if f.endswith('.txt') and f != 'prompts.txt' and not f.startswith('generate_prompts_txt')]
prompt_files.sort()

with open(os.path.join(baroque_dir, 'prompts.txt'), 'w') as out:
    for fname in prompt_files:
        with open(os.path.join(baroque_dir, fname), 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            out.write(prompt + '\n')
print(f"Wrote {len(prompt_files)} prompts to prompts.txt")

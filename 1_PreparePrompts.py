import csv
import json
from pathlib import Path

# Define the prompt template
PROMPT_TEMPLATE = """আপনি একজন লোককথা-কথক।

বাংলাদেশের {culture} সংস্কৃতির প্রেক্ষাপটে, নিচের কাহিনি ধরণ অনুসারে একটি লোকজ গল্প লিখুন।

গল্পটি প্রমিত বাংলায় লিখবেন। তবে সংস্কৃতিগতভাবে প্রাসঙ্গিক হলে আঞ্চলিক শব্দ, বিশ্বাস, রীতি, পরিবেশ, প্রতীক বা সামাজিক আচরণ স্বাভাবিকভাবে ব্যবহার করতে পারেন। যেসব বিষয়ের বিষয়ে নিশ্চিত নন, সেগুলো কল্পিতভাবে নির্দিষ্ট করে উপস্থাপন করবেন না।

গল্পে আধুনিক প্রযুক্তি, আধুনিক রাষ্ট্রব্যবস্থা, নির্দিষ্ট সাল বা সমসাময়িক শব্দ ব্যবহার করবেন না। গল্পটি ঐতিহ্যগত সামাজিক প্রেক্ষাপটে আবদ্ধ থাকবে।

গল্পের দৈর্ঘ্য আনুমানিক ৫০০–৭০০ শব্দ।

কাহিনি ধরণ:
{story_scenario}"""

def load_cultures(file_path):
    """Load cultures from CSV file."""
    cultures = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cultures.append({
                'culture': row['Cultures'],
                'region': row['Region']
            })
    return cultures

def load_scenarios(file_path):
    """Load scenarios from CSV file."""
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append({
                'story_type': row['Story_Type'],
                'scenario': row['Scenario']
            })
    return scenarios

def generate_prompts():
    """Generate all prompt combinations."""
    # Load data
    cultures = load_cultures('plans/cultures.csv')
    scenarios = load_scenarios('plans/scenarios.csv')
    
    # Generate prompts
    prompts = []
    prompt_id = 1
    
    for culture_data in cultures:
        for scenario_data in scenarios:
            prompt_text = PROMPT_TEMPLATE.format(
                culture=culture_data['culture'],
                story_scenario=scenario_data['scenario']
            )
            
            prompt_obj = {
                'prompt_id': f"P{prompt_id:03d}",
                'culture': culture_data['culture'],
                'region': culture_data['region'],
                'story_type': scenario_data['story_type'],
                'scenario': scenario_data['scenario'],
                'prompt': prompt_text
            }
            
            prompts.append(prompt_obj)
            prompt_id += 1
    
    return prompts

def main():
    """Main function to generate and save prompts."""
    print("Loading cultures and scenarios...")
    prompts = generate_prompts()
    
    print(f"Generated {len(prompts)} prompts")
    
    # Save to JSON
    output_file = 'prompts.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    print(f"Prompts saved to {output_file}")
    
    # Print sample
    print("\nSample prompt:")
    print(f"ID: {prompts[0]['prompt_id']}")
    print(f"Culture: {prompts[0]['culture']}")
    print(f"Story Type: {prompts[0]['story_type']}")
    print(f"\nPrompt:\n{prompts[0]['prompt'][:200]}...")

if __name__ == "__main__":
    main()

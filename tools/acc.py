import csv

def evaluate_accuracy(ground_truth_csv, predictions_csv, model_name="Model"):
    # 1. Load the "Ground Truth" (Your Korean Predictions)
    gt_data = {}
    with open(ground_truth_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data[row['Filename']] = {
                'Year': row['Year'],
                'Month': row['Month'],
                'Day': row['Day'],
                'Raw': row.get('Raw_String', 'N/A')
            }

    # 2. Track our metrics and failures
    total_evaluated = 0
    total_correct = 0
    ambiguous_total = 0
    ambiguous_correct = 0
    
    # List to store our failed cases
    failed_cases = []

    # 3. Read the Predictions and compare
    with open(predictions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename']
            
            # Skip if for some reason the file isn't in both lists
            if filename not in gt_data:
                continue

            total_evaluated += 1
            gt = gt_data[filename]
            raw_string = gt['Raw']
            
            # Check if Year, Month, and Day match perfectly
            is_correct = (
                row['Year'] == gt['Year'] and 
                row['Month'] == gt['Month'] and 
                row['Day'] == gt['Day']
            )

            if is_correct:
                total_correct += 1
            else:
                # Store the failure details
                failed_cases.append({
                    'Filename': filename,
                    'Raw': raw_string,
                    'Expected': f"{gt['Year']}-{gt['Month']}-{gt['Day']}",
                    'Predicted': f"{row['Year']}-{row['Month']}-{row['Day']}"
                })

            # 4. Identify if this was a tricky "Ambiguous" date (e.g., 05/06)
            try:
                gt_m = int(gt['Month'])
                gt_d = int(gt['Day'])
                if gt_m <= 12 and gt_d <= 12 and gt_m != gt_d:
                    ambiguous_total += 1
                    if is_correct:
                        ambiguous_correct += 1
            except ValueError:
                pass # Handles cases where the output was "None" safely

    # 5. Calculate Final Percentages
    accuracy = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
    amb_accuracy = (ambiguous_correct / ambiguous_total) * 100 if ambiguous_total > 0 else 0

    # 6. Print the Results for the Paper
    print("\n" + "="*60)
    print(f"   ABLATION STUDY RESULTS: {model_name.upper()}")
    print("="*60)
    print(f"Total Images Evaluated: {total_evaluated}")
    print(f"Overall Accuracy:       {accuracy:.2f}%  ({total_correct}/{total_evaluated})")
    print("-" * 40)
    print(f"Total Ambiguous Dates:  {ambiguous_total}")
    print(f"Ambiguity Accuracy:     {amb_accuracy:.2f}%  ({ambiguous_correct}/{ambiguous_total})")
    print("="*60)
    
    # 7. Print the Failed Cases
    if failed_cases:
        print(f"\n--- FAILED CASES FOR {model_name.upper()} ({len(failed_cases)} total) ---")
        for fail in failed_cases:
            print(f"[{fail['Filename']}] Raw: '{fail['Raw']}' | Expected: {fail['Expected']} | Guessed: {fail['Predicted']}")
    else:
        print(f"\n--- NO FAILED CASES FOR {model_name.upper()}! PERFECT SCORE! ---")
        
    print("\n")


# ==========================================
# RUN THE EVALUATIONS
# ==========================================

# 1. Evaluate the original baseline parser
evaluate_accuracy(
    'korean_baseline_predictions.csv', 
    'baseline_predictions.csv', 
    model_name="Heuristic Baseline"
)

# 2. Evaluate the new CBR voting parser
evaluate_accuracy(
    'korean_baseline_predictions.csv', 
    'cbr_predictions.csv', 
    model_name="CBR Voting Module"
)
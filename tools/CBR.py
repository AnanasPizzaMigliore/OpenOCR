import re
from datetime import datetime
from collections import defaultdict

class CaseBasedDateParser:
    def __init__(self, enable_decay=False, decay_factor=0.995):
        self.case_base = defaultdict(float)
        self.enable_decay = enable_decay
        self.decay_factor = decay_factor

        self.noise_patterns = [
            (r'[\[\]\{\}]', ''),                   
            (r'\|', '-'),                          
            (r'[\u4e00-\u9fa5]', ' '),
            (r'[^a-zA-Z0-9\s\.\-\/:]', ' '),
            (r'(\d{4})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{1,2})', r'\1-\2-\3'),
            (r'(?<!\d)(\d{1,2})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{4})', r'\1-\2-\3'),
        ]

        self.month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        cleaned = text.strip()
        for pattern, repl in self.noise_patterns:
            cleaned = re.sub(pattern, repl, cleaned)
        return cleaned.strip()

    def decay_case_base(self):
        if self.enable_decay:
            for k in self.case_base:
                self.case_base[k] *= self.decay_factor

    def parse(self, text):
        if not text: return None
        clean_text = self.clean_text(text)
        if len(clean_text) < 4 or not any(c.isdigit() for c in clean_text): return None

        candidates = []
        now = datetime.now()
        fallback_year = now.year

        self.decay_case_base()

        # ==================================================
        # GENERATE CANDIDATES & STRUCTURAL TAGS
        # (Updated to accept spaces as valid separators)
        # ==================================================
        
        # 1. TWO-PART DATES (e.g., 04.13, 10.01, 09 12)
        m2 = re.match(r'^\s*(\d{1,2})[\s\.\-/]+(\d{1,2})\s*$', clean_text)
        if m2:
            a, b = int(m2.group(1)), int(m2.group(2))
            if 1 <= a <= 12 and 1 <= b <= 31: 
                try: candidates.append((datetime(fallback_year, a, b), 'MONTH_FIRST_SHORT'))
                except ValueError: pass
            if 1 <= b <= 12 and 1 <= a <= 31: 
                try: candidates.append((datetime(fallback_year, b, a), 'DAY_FIRST_SHORT'))
                except ValueError: pass

        # 2. PARTIAL NUMERIC DATES (e.g., 05.2024, 2024.05, 09 2020)
        m2_year = re.match(r'^\s*(\d{1,4})[\s\.\-/]+(\d{1,4})\s*$', clean_text)
        if m2_year and not candidates: # Only if it didn't generate valid 2-digit day/month candidates
            a, b = int(m2_year.group(1)), int(m2_year.group(2))
            if 2015 <= a <= 2035 and 1 <= b <= 12: candidates.append((datetime(a, b, 1), 'YEAR_FIRST'))
            if 2015 <= b <= 2035 and 1 <= a <= 12: candidates.append((datetime(b, a, 1), 'MONTH_FIRST'))

        # 3. THREE-PART DATES (e.g., 21.05.23, 2021.05.23, 21 05 23)
        m3 = re.match(r'^\s*(\d{1,4})[\s\.\-/]+(\d{1,2})[\s\.\-/]+(\d{1,4})\s*$', clean_text)
        if m3:
            a, b, c = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
            
            if a > 1000: # YYYY.MM.DD
                if 1 <= b <= 12 and 1 <= c <= 31 and 2015 <= a <= 2035: 
                    try: candidates.append((datetime(a, b, c), 'YEAR_FIRST'))
                    except ValueError: pass
            elif c > 1000: # DD.MM.YYYY or MM.DD.YYYY
                if 1 <= b <= 12 and 1 <= a <= 31 and 2015 <= c <= 2035: 
                    try: candidates.append((datetime(c, b, a), 'DAY_FIRST'))
                    except ValueError: pass
                if 1 <= a <= 12 and 1 <= b <= 31 and 2015 <= c <= 2035: 
                    try: candidates.append((datetime(c, a, b), 'MONTH_FIRST'))
                    except ValueError: pass
            else: # YY.MM.DD or DD.MM.YY
                y_a = a + 2000
                y_c = c + 2000
                if 1 <= b <= 12 and 1 <= c <= 31 and 2015 <= y_a <= 2035: 
                    try: candidates.append((datetime(y_a, b, c), 'YEAR_FIRST'))
                    except ValueError: pass
                if 1 <= b <= 12 and 1 <= a <= 31 and 2015 <= y_c <= 2035: 
                    try: candidates.append((datetime(y_c, b, a), 'DAY_FIRST'))
                    except ValueError: pass
                if 1 <= a <= 12 and 1 <= b <= 31 and 2015 <= y_c <= 2035: 
                    try: candidates.append((datetime(y_c, a, b), 'MONTH_FIRST'))
                    except ValueError: pass

        # 4. TEXT MONTH FORMATS (e.g., 10 MAY 2021)
        upper_text = clean_text.upper()
        text_month_regexes = [
            (r'^\s*(\d{1,2})[\s\.\/\-]+([A-Z]{3,4})[\s\.\/\-]+(\d{2,4})\s*$', 'DAY_FIRST'), 
            (r'^\s*([A-Z]{3,4})[\s\.\/\-]+(\d{1,2})[\s\.\/\-]+(\d{2,4})\s*$', 'MONTH_FIRST'),
            (r'^\s*(\d{4})[\s\.\/\-]+([A-Z]{3,4})[\s\.\/\-]+(\d{1,2})\s*$', 'YEAR_FIRST'),
            (r'^\s*(\d{4})[\s\.\/\-]+(\d{1,2})[\s\.\/\-]+([A-Z]{3,4})\s*$', 'YEAR_FIRST')
        ]
        
        for reg, tag_type in text_month_regexes:
            m_match = re.match(reg, upper_text)
            if m_match:
                groups = m_match.groups()
                mon_str = next(g for g in groups if not g.isdigit())
                nums = [int(g) for g in groups if g.isdigit()]
                
                if mon_str in self.month_map:
                    mon = self.month_map[mon_str]
                    if tag_type == 'YEAR_FIRST': y, d = nums[0], nums[1]
                    else: d, y = nums[0], nums[1]
                        
                    if y < 100: y += 2000
                    if 2015 <= y <= 2035:
                        try: candidates.append((datetime(y, mon, d), tag_type))
                        except ValueError: pass

        # 4.5. PARTIAL TEXT MONTHS (e.g., NOV 2021)
        partial_text_regexes = [
            (r'^\s*([A-Z]{3,4})[\s\.\/\-]+(\d{2,4})\s*$', 'MONTH_FIRST'),
            (r'^\s*(\d{2,4})[\s\.\/\-]+([A-Z]{3,4})\s*$', 'YEAR_FIRST')
        ]
        
        for reg, tag_type in partial_text_regexes:
            m_match = re.match(reg, upper_text)
            if m_match:
                groups = m_match.groups()
                mon_str = next(g for g in groups if not g.isdigit())
                y_str = next(g for g in groups if g.isdigit())
                
                if mon_str in self.month_map:
                    mon = self.month_map[mon_str]
                    y = int(y_str)
                    if y < 100: y += 2000
                    if 2015 <= y <= 2035:
                        try: candidates.append((datetime(y, mon, 1), tag_type))
                        except ValueError: pass

        # 5. DENSE NUMERIC (e.g., 210523)
        digits_only = re.sub(r'\D', '', clean_text)
        dense = []
        is_partial = re.match(r'^\s*(\d{1,2}[\s\.\-/]+\d{4}|\d{4}[\s\.\-/]+\d{1,2})\s*$', clean_text)
        
        has_letters = bool(re.search(r'[A-Za-z]', clean_text))
        
        if not is_partial and not has_letters:
            if len(digits_only) == 4:
                dense = [('%m%d', 'MONTH_FIRST_SHORT'), ('%d%m', 'DAY_FIRST_SHORT')]
            elif len(digits_only) == 6:
                dense = [('%y%m%d', 'YEAR_FIRST'), ('%d%m%y', 'DAY_FIRST'), ('%m%d%y', 'MONTH_FIRST')]
            elif len(digits_only) == 8:
                dense = [('%Y%m%d', 'YEAR_FIRST'), ('%d%m%Y', 'DAY_FIRST'), ('%m%d%Y', 'MONTH_FIRST')]

        for fmt, tag in dense:
            try:
                dt = datetime.strptime(digits_only, fmt)
                if dt.year == 1900: dt = dt.replace(year=fallback_year)
                if 2015 <= dt.year <= 2035: candidates.append((dt, tag))
            except ValueError:
                continue

        if not candidates:
            return None

        # ==================================================
        # THE CBR VOTING ENGINE
        # ==================================================
        dt_to_tags = defaultdict(list)
        for dt, tag in candidates:
            if tag not in dt_to_tags[dt]:
                dt_to_tags[dt].append(tag)

        # CASE 1: UNAMBIGUOUS
        if len(dt_to_tags) == 1:
            best_dt = next(iter(dt_to_tags))
            for tag in dt_to_tags[best_dt]:
                self.case_base[tag] += 1
            return best_dt 

        # CASE 2: AMBIGUOUS
        dt_scores = {}
        for dt, tags in dt_to_tags.items():
            dt_scores[dt] = sum(self.case_base[tag] for tag in tags)

        best_dt = max(dt_scores, key=dt_scores.get)
        sorted_scores = sorted(dt_scores.values(), reverse=True)

        if sorted_scores[0] > 0 and (len(sorted_scores) == 1 or sorted_scores[0] > sorted_scores[1]):
            for tag in dt_to_tags[best_dt]:
                self.case_base[tag] += 1
            return best_dt

        # CASE 3: FALLBACK
        def temporal_score(dt):
            return abs((dt - now).days)

        best_dt = min(dt_to_tags.keys(), key=temporal_score)
        for tag in dt_to_tags[best_dt]:
            self.case_base[tag] += 1

        return best_dt

def parse_dates_from_file(input_filepath, output_filepath):
    parser = CaseBasedDateParser(enable_decay=False)
    results = []

    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip(): continue

        parts = line.strip().split('\t', 1)
        if len(parts) != 2: continue

        filename, raw = parts
        parsed_dt = parser.parse(raw)

        if parsed_dt:
            year = parsed_dt.strftime('%Y')
            month = parsed_dt.strftime('%m')
            day = parsed_dt.strftime('%d')
        else:
            year, month, day = "None", "None", "None"

        results.append(f"{filename},{raw},{year},{month},{day}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("Filename,Raw_String,Year,Month,Day\n")
        f.write("\n".join(results))

    print(f"Done! Parsed {len(results)} entries using the CBR Voting Module.")
    print("Final Format Memory Votes:")
    for tag, score in parser.case_base.items():
        print(f" - {tag}: {score}")
    print(f"Saved to: {output_filepath}")

if __name__ == "__main__":
    parse_dates_from_file(
        '/scratch/penghao/datasets/Date-Real/eval/labels_fixed.txt',
        'cbr_predictions.csv'
    )
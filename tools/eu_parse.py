import re
from datetime import datetime
from dateutil import parser as date_parser

class DateParser:
    def __init__(self):
        self.noise_patterns = [
            (r'[|\[\]\{\}]', ''),
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
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        for pattern, replacement in self.noise_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()

    def parse(self, text):
        if not text:
            return None

        clean_text = self.clean_text(text)
        if len(clean_text) < 4 or not any(c.isdigit() for c in clean_text):
            return None

        now = datetime.now()

        # ==================================================
        # 1. TWO-PART DATES (Strict South Korean Preference: MM.DD)
        # ==================================================
        two_part_match = re.match(r'^\s*(\d{1,2})[.\-/](\d{1,2})\s*$', clean_text)
        if two_part_match:
            a, b = int(two_part_match.group(1)), int(two_part_match.group(2))
            
            # PREFERENCE 1: South Korean (MM.DD) 
            # e.g., "04.13" -> Month 4, Day 13
            if 1 <= a <= 12 and 1 <= b <= 31:
                try:
                    return datetime(now.year, a, b)
                except ValueError:
                    pass
            
            # PREFERENCE 2: Fallback (DD.MM) 
            # Only triggers if Preference 1 is physically impossible (e.g., 25.11)
            if 1 <= a <= 31 and 1 <= b <= 12:
                try:
                    return datetime(now.year, b, a)
                except ValueError:
                    pass

        # ==================================================
        # 1.5. MM.YYYY / YYYY.MM (PARTIAL DATES -> FIRST DAY)
        # ==================================================
        ym_match = re.match(r'^\s*(\d{4})[.\-/](\d{1,2})\s*$', clean_text)
        if ym_match:
            y, m = int(ym_match.group(1)), int(ym_match.group(2))
            if 1 <= m <= 12 and 2000 <= y <= 2035:
                return datetime(y, m, 1)

        my_match = re.match(r'^\s*(\d{1,2})[.\-/](\d{4})\s*$', clean_text)
        if my_match:
            m, y = int(my_match.group(1)), int(my_match.group(2))
            if 1 <= m <= 12 and 2000 <= y <= 2035:
                return datetime(y, m, 1)

        # ==================================================
        # 2. THREE-PART DATES (Strict South Korean YY.MM.DD)
        # ==================================================
        yy_match = re.match(r'^\s*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{1,2})\s*$', clean_text)
        if yy_match:
            part1, part2, part3 = int(yy_match.group(1)), int(yy_match.group(2)), int(yy_match.group(3))
            
            # PREFERENCE 1: YY.MM.DD (Korean Standard)
            y1 = part1 + 2000 if part1 < 100 else part1
            if 1 <= part2 <= 12 and 1 <= part3 <= 31:
                try:
                    return datetime(y1, part2, part3)
                except ValueError:
                    pass
                    
            # PREFERENCE 2: DD.MM.YY (Imported European goods fallback)
            y3 = part3 + 2000 if part3 < 100 else part3
            if 1 <= part2 <= 12 and 1 <= part1 <= 31:
                try:
                    return datetime(y3, part2, part1)
                except ValueError:
                    pass

        # ==================================================
        # 3. YEAR-FIRST FORMATS (YYYY.MM.DD)
        # ==================================================
        ymd_match = re.match(r'^\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\s*$', clean_text)
        if ymd_match:
            y, m, d = int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3))
            try:
                return datetime(y, m, d)
            except ValueError:
                pass

        # ==================================================
        # 4. TEXT MONTH FORMATS (STRICT, RETURN EARLY)
        # ==================================================
        upper_text = clean_text.upper()
        
        for reg in [r'^\s*(\d{1,2})\s+([A-Z]{3,4})\s+(\d{2})\s*$', 
                    r'^\s*([A-Z]{3,4})\s+(\d{1,2})\s+(\d{2})\s*$',
                    r'^\s*([A-Z]{3,4})[\/\-](\d{1,2})[\/\-](\d{2})\s*$']:
            m_match = re.match(reg, upper_text)
            if m_match:
                groups = m_match.groups()
                # Simplified text month catching
                for g in groups:
                    if g in self.month_map:
                        mon = self.month_map[g]
                        nums = [int(x) for x in groups if x != g]
                        y = nums[1] + 2000
                        d = nums[0]
                        try:
                            return datetime(y, mon, d)
                        except:
                            pass

        # ==================================================
        # 5. DENSE NUMERIC (Korean preference)
        # ==================================================
        candidates = []
        digits_only = re.sub(r'\D', '', clean_text)

        dense_formats = []
        if len(digits_only) == 4:
            dense_formats = ['%m%d', '%d%m'] 
        elif len(digits_only) == 6:
            dense_formats = ['%y%m%d', '%d%m%y']
        elif len(digits_only) == 8:
            dense_formats = ['%Y%m%d', '%d%m%Y']

        for fmt in dense_formats:
            try:
                dt = datetime.strptime(digits_only, fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=now.year)
                candidates.append(dt)
            except ValueError:
                continue

        # ==================================================
        # 6. DATEUTIL (LAST RESORT)
        # ==================================================
        for yfirst, dfirst in [(True, False), (False, True)]:
            try:
                dt = date_parser.parse(clean_text, fuzzy=True, yearfirst=yfirst, dayfirst=dfirst)
                if dt.year < 1901:
                    dt = dt.replace(year=now.year)
                candidates.append(dt)
            except Exception:
                continue

        if not candidates:
            return None

        # ==================================================
        # 7. SELECT BEST (closest to today)
        # ==================================================
        unique = {}
        for dt in candidates:
            unique[dt.date()] = dt

        best = None
        min_diff = float('inf')

        for dt in unique.values():
            if not (2000 <= dt.year <= 2035):
                continue
            diff = abs((dt.date() - now.date()).days)
            if diff < min_diff:
                min_diff = diff
                best = dt

        return best


# ==========================================
# FILE PROCESSING
# ==========================================
def parse_dates_from_file(input_filepath, output_filepath):
    parser = DateParser()
    results = []

    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split('\t', 1)
        if len(parts) != 2:
            continue

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

    print(f"Done! Parsed {len(results)} entries.")
    print(f"Saved to: {output_filepath}")


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    parse_dates_from_file(
        '/scratch/penghao/datasets/Date-Real/eval/labels_fixed.txt',
        'korean_baseline_predictions.csv'
    )
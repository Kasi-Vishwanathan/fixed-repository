"""
Evaluation metrics to measure functional correctness of traces.
"""
text_identifier_num = 0
gold_identifier_num = 0
correct_identifier_num = 0

def get_output_from_trace(text):
    output_list = []
    parse_loc = []
    start_len = 0
    # Find all <line> tags
    while True:
        num = text.find("<line>", start_len)
        if num == -1:
            break
        parse_loc.append(num)
        start_len = num + len("<line>")  # Skip past the found tag
    # Find all <output> tags
    start_len = 0
    while True:
        num = text.find("<output>", start_len)
        if num == -1:
            break
        parse_loc.append(num)
        start_len = num + len("<output>")  # Skip past the found tag
    # Add boundaries
    parse_loc.extend([0, len(text)])
    parse_loc = sorted(set(parse_loc))  # Deduplicate and sort
    
    for i in range(1, len(parse_loc)):
        start = parse_loc[i-1]
        end = parse_loc[i]
        segment = text[start:end]
        # Handle last segment
        if i == len(parse_loc)-1:
            if "</state>" not in segment:
                continue
        # Extract output content
        if "<output>" in segment:
            output_start = segment.find("<output>") + len("<output>")
            output_end = segment.find("</output>", output_start)
            if output_end != -1:
                output = segment[output_start:output_end]
                output_list.append(output)
    return output_list
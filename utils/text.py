def clean_transcript(trans, labels, upper=True, space_symbol="|"):
    cleaned = ""
    trans = trans.replace(" ", space_symbol)
    for char in trans:
        if char in labels:
            cleaned += char
    if upper:
        cleaned = cleaned.upper()
    return cleaned
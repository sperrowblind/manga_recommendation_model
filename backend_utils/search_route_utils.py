from difflib import SequenceMatcher

def search_find_matches(results, search_term):
    matches = []
    for title, rating, source, model_version in results:
        ratio = SequenceMatcher(a=title.lower(), b=search_term.lower()).ratio()
        if ratio >= 0.65 and rating is not None:
            matches.append((title, rating, source, model_version, ratio))
    return matches


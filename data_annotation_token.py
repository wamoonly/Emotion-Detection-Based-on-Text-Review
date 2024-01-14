sentence = ['This', 'newly', 'renovated', 'place', 'requires', 'waiting', 'in', 'a', 'long', 'queue', '.', 'You', 'should', 'go', 'on', 'weekdays', 'if', 'you', 'do', 'not', 'want', 'to', 'wait', 'in', 'a', 'long', 'line', '.']

aspect_terms =  ['queue']
tags = [0] * len(sentence)

for aspect_term in aspect_terms:
    aspect_tokens = aspect_term.split()
    for i, token in enumerate(sentence):
        if token == aspect_tokens[0]:
            tags[i] = 1
            for j in range(1, len(aspect_tokens)):
                if i + j < len(sentence) and sentence[i + j] == aspect_tokens[j]:
                    tags[i + j] = 2

print(tags)
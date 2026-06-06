from collections import Counter
from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            
            # count adjacent pair frequencies
            pairs = Counter(zip(tokens, tokens[1:]))

            if not pairs:
                break

            # max count, tiebreak = lexicographically smallest pair
            best = min(pairs, key=lambda p: (-pairs[p], p))
            merges.append([best[0], best[1]])

            # merge all non-overlapping occurrences left to right
            new_tokens = []
            i, n = 0, len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges

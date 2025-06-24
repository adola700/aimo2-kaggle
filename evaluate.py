class evaluate_all:
    def __init__(self, answers, lengths, targets):
        self.answers = answers
        self.lengths = lengths
        self.targets = targets
        self.N = len(answers)

        self.answers_flatten = sum(self.answers, [])
        self.lengths_flatten = sum(self.lengths, [])

    def _filter_responses(self, filtered_length):
        for i in range(self.N):
            for j in range(len(self.answers[i])):
                if self.lengths[i][j]>filtered_length:
                    self.answers[i][j] = -1000
                    self.lengths[i][j] = filtered_length
                else:
                    pass

    def _helper_maj_vote(self, lst):
        """
        Finds the element with the highest frequency in a list.
    
        Args:
            lst (list): A list of numbers.
    
        Returns:
            int or None: The element with the highest frequency, or None if the list is empty.
        """
        from collections import Counter    
        lst = [i for i in lst if i != -1000]
        
        if len(lst) == 0:
            return -1000
    
        # Count the occurrences of each element in the list
        counts = Counter(lst)
    
        # Find the element with the highest frequency
        most_common = counts.most_common(1)
        return most_common[0][0] if most_common else None
        
    def _pass_1(self):
        count = 0
        for i in range(self.N):
                count += sum([1 for ans in self.answers[i] if ans==self.targets[i] ])/len(self.answers[i])
        return count/self.N*100

    def _atleast_one(self):
        count = 0
        for i in range(self.N):
            if self.targets[i] in self.answers[i]:
                count += 1
        return count/self.N*100
    
    def _maj_vote(self):
        count = 0
        for i in range(self.N):
            if self.targets[i] == self._helper_maj_vote(self.answers[i]):
                count += 1
        return count/self.N*100

    def print_all_metrics(self):
        print("\nMetrics for full context:\n")
        
        print(f"pass@1: {self._pass_1()}")
        print(f"maj_vote: {self._maj_vote()}")
        print(f"atleast_one: {self._atleast_one()}")

        self._filter_responses(12000)
        print("\nMetrics for 12k:\n")

        print(f"pass@1: {self._pass_1()}")
        print(f"maj_vote: {self._maj_vote()}")
        print(f"atleast_one: {self._atleast_one()}")

        self._filter_responses(8000)
        print("\nMetrics for 8k:\n")

        print(f"pass@1: {self._pass_1()}")
        print(f"maj_vote: {self._maj_vote()}")
        print(f"atleast_one: {self._atleast_one()}")

        

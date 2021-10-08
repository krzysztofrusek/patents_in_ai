import unittest
import data

class DataTestCase(unittest.TestCase):
    def test_cpcai(self):
        data.AICPC().ai_cpc

    def test_ai_counts(self):
        clean_df = data.load_clean()
        aicpc = data.AICPC()
        counts = aicpc.counts(clean_df)
        self.assertTrue(sum(counts.values())>clean_df.shape[0])



if __name__ == '__main__':
    unittest.main()

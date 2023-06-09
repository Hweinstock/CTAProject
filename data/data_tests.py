import unittest

from read_data import read_in_chunked_data

CUTOFF = 10
MAX_INPUT = 512

class TestData(unittest.TestCase):
    full_df = None

    def setUp(self):
        data_source = 'processed_headline_data/'
        self.full_df = read_in_chunked_data(data_source, '')
        self.full_df['text_len'] = self.full_df['text'].map(lambda x: len(x.split()))

    def test_structure(self):
        self.assertNotEqual(len(self.full_df.index), 0)
        desired_columns = ['text','stock', 'date', 'close' ,'volume','next_close', 'next_volume','label']
        for col in desired_columns:
            self.assertTrue(col in self.full_df.columns)
    
    def test_stock_counts(self):
        stock_counts = self.full_df['stock'].value_counts()
        self.assertGreaterEqual(stock_counts.min(), CUTOFF)
    
    def test_missing_values(self):
        self.assertFalse(self.full_df.isnull().values.any())
    
    def test_max_length(self):
        self.assertLessEqual(self.full_df['text_len'].max(), 512)
    
    def test_empty_text(self):
        self.assertGreater(self.full_df['text_len'].min(), 0)

    def test_somewhat_balanced(self):
        total = len(self.full_df.index)
        increase_pct = len(self.full_df[self.full_df['label'] == 0].index) / total
        decrease_pct = len(self.full_df[self.full_df['label'] == 1].index) / total
        neutral_pct = len(self.full_df[self.full_df['label'] == 2].index) / total
        print(increase_pct, decrease_pct, neutral_pct)
        cutoff = 0.28 
        self.assertGreater(increase_pct, cutoff)
        self.assertGreater(decrease_pct, cutoff)
        self.assertGreater(neutral_pct, cutoff)

if __name__ == '__main__':
    unittest.main()

from datasets import load_dataset
import tqdm

dataset = load_dataset('bookcorpus')['train']

total = len(dataset)
fp = open('books.txt', 'w', encoding='utf-8')
for i in tqdm.tqdm(range(total)):
    fp.write(dataset[i]['text'])
    fp.write('\n')
fp.close()

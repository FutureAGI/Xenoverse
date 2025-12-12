import pickle

if __name__=='__main__':
    with open('task.pkl', 'rb') as f:
        task = pickle.load(f)
    print(task)
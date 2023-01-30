import pandas as pd


def merge_train_plans():
    
    train_data = pd.DataFrame()
    
    for idx in range(0, 20):
        csv_file = f'train/train_plan_part{idx}.csv'
        
        df = pd.read_csv(csv_file)
        
        train_data = pd.concat((train_data, df))
        
        print(len(train_data))
        
        save_path = f'train_merged/train_plan_{len(train_data)}.csv'
        
        train_data.to_csv(save_path)
        
    for sample in [500, 1000, 2000]:
        sample_data = train_data.iloc[:sample]
        print(len(sample_data))
        
        save_path = f'train_merged/train_plan_{len(sample_data)}.csv'
        
        sample_data.to_csv(save_path)
        
    
if __name__ == '__main__':
    
    merge_train_plans()
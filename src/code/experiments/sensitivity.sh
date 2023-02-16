cd /home/atul/Desktop/Query-Cost-Estimation-with-Deep-Learning
source venv/bin/activate
cd src

python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_xgb_100000 --num-models=5 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=90000 --name=tree_xgb_90000 --num-models=5 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=80000 --name=tree_xgb_80000 --num-models=5 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=70000 --name=tree_xgb_70000 --num-models=4 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=60000 --name=tree_xgb_60000 --num-models=4 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=50000 --name=tree_xgb_50000 --num-models=4 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=40000 --name=tree_xgb_40000 --num-models=3 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=30000 --name=tree_xgb_30000 --num-models=3 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=20000 --name=tree_xgb_20000 --num-models=2 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=10000 --name=tree_xgb_10000 --num-models=1 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=5000 --name=tree_xgb_5000 --num-models=1 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=2000 --name=tree_xgb_2000 --num-models=1 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=1000 --name=tree_xgb_1000 --num-models=1 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference


python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_lgbm_100000 --num-models=5 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=90000 --name=tree_lgbm_90000 --num-models=5 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=80000 --name=tree_lgbm_80000 --num-models=5 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=70000 --name=tree_lgbm_70000 --num-models=4 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=60000 --name=tree_lgbm_60000 --num-models=4 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=50000 --name=tree_lgbm_50000 --num-models=4 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=40000 --name=tree_lgbm_40000 --num-models=3 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=30000 --name=tree_lgbm_30000 --num-models=3 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=20000 --name=tree_lgbm_20000 --num-models=2 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=10000 --name=tree_lgbm_10000 --num-models=1 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=5000 --name=tree_lgbm_5000 --num-models=1 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=2000 --name=tree_lgbm_2000 --num-models=1 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=1000 --name=tree_lgbm_1000 --num-models=1 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

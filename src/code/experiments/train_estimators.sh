cd /home/atul/Desktop/Query-Cost-Estimation-with-Deep-Learning
source venv/bin/activate
cd src


python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_lgbm --num-models=5 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --save

python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_lgbm --num-models=5 --method=lgbm --depth=16 --learning-rate=0.6 --n_estimators=200 --fast-inference

python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_xgb --num-models=5 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --save

python3 -m code.train.train_tree_gbm --dataset=imdb --size=100000 --name=tree_xgb --num-models=5 --method=xgb --depth=16 --learning-rate=0.1 --n_estimators=100 --fast-inference

python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_nn --epochs=30 --method=tree_nn

python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_lstm --epochs=30 --method=tree_lstm

python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_gru --epochs=30 --method=tree_gru

python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_attn --epochs=30 --method=tree_attn
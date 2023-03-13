cd /home/atul/Desktop/Query-Cost-Estimation-with-Deep-Learning
source venv/bin/activate
cd src

python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_nn_100000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=90000 --name=tree_nn_90000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=80000 --name=tree_nn_80000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=70000 --name=tree_nn_70000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=60000 --name=tree_nn_60000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=50000 --name=tree_nn_50000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=40000 --name=tree_nn_40000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=30000 --name=tree_nn_30000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=20000 --name=tree_nn_20000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=10000 --name=tree_nn_10000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=5000 --name=tree_nn_5000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=2000 --name=tree_nn_2000 --epochs=30 --method=tree_nn
python3 -m code.train.train_tree_nn --dataset=imdb --size=1000 --name=tree_nn_1000 --epochs=30 --method=tree_nn


python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_lstm_100000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=90000 --name=tree_lstm_90000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=80000 --name=tree_lstm_80000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=70000 --name=tree_lstm_70000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=60000 --name=tree_lstm_60000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=50000 --name=tree_lstm_50000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=40000 --name=tree_lstm_40000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=30000 --name=tree_lstm_30000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=20000 --name=tree_lstm_20000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=10000 --name=tree_lstm_10000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=5000 --name=tree_lstm_5000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=2000 --name=tree_lstm_2000 --epochs=30 --method=tree_lstm
python3 -m code.train.train_tree_nn --dataset=imdb --size=1000 --name=tree_lstm_1000 --epochs=30 --method=tree_lstm


python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_gru_100000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=90000 --name=tree_gru_90000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=80000 --name=tree_gru_80000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=70000 --name=tree_gru_70000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=60000 --name=tree_gru_60000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=50000 --name=tree_gru_50000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=40000 --name=tree_gru_40000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=30000 --name=tree_gru_30000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=20000 --name=tree_gru_20000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=10000 --name=tree_gru_10000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=5000 --name=tree_gru_5000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=2000 --name=tree_gru_2000 --epochs=30 --method=tree_gru
python3 -m code.train.train_tree_nn --dataset=imdb --size=1000 --name=tree_gru_1000 --epochs=30 --method=tree_gru


python3 -m code.train.train_tree_nn --dataset=imdb --size=100000 --name=tree_attn_100000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=90000 --name=tree_attn_90000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=80000 --name=tree_attn_80000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=70000 --name=tree_attn_70000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=60000 --name=tree_attn_60000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=50000 --name=tree_attn_50000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=40000 --name=tree_attn_40000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=30000 --name=tree_attn_30000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=20000 --name=tree_attn_20000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=10000 --name=tree_attn_10000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=5000 --name=tree_attn_5000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=2000 --name=tree_attn_2000 --epochs=30 --method=tree_attn
python3 -m code.train.train_tree_nn --dataset=imdb --size=1000 --name=tree_attn_1000 --epochs=30 --method=tree_attn


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

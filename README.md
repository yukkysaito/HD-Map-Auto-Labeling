# HD-Map-Auto-Labeling — rosbag から HD Map 学習データを自動生成する

rosbag（カメラ・自己位置・公式キャリブレーション）と LiDAR 点群地図から、マルチカメラ・複数フレーム入力の
オンライン HD Map 推定モデル用の教師データを作る。教師データの中身は次の通り。
- ラスタ（車道・白線・横断歩道・縁石・歩道…）
- 3D ベクトル（車線の線、横断歩道、縁石、道路境界、**右左折の矢印と進行可能方向**）

**HD Map（lanelet2 / ベクタ地図）は使わない。** 平面路面も仮定しないので、坂・立体交差・高架の下でも成り立つ。

## 必要なもの
- 入力
  - rosbag（mcap、ROS 2）: カメラ画像（CompressedImage）と camera_info、自己位置（Odometry、地図座標の base_link）、/tf_static。
    車速（VelocityReport）は任意。トピック名とカメラは `configs/default.yaml` で設定する（既定は Autoware の名前）。
  - LiDAR の点群地図（.pcd）。自己位置と同じ地図座標のもの。
- 環境: Linux、CUDA の使える NVIDIA GPU（12 GB で動作確認）、Python 3.11。セットアップは [env/README.md](env/README.md)。
  セグメンテーションのモデル（Mask2Former, Mapillary Vistas）は初回に Hugging Face から自動で取得される。

## 使い方
```bash
./autolabel run --map /data/pointcloud_map.pcd --logs "/data/logs/*" --name my_area   # bag → 学習データ + デバッグ出力
./autolabel status --name my_area                                                      # 何が終わり、何が失敗したか
./autolabel test                                                                       # 単体テスト
```
- `--logs`: glob。ディレクトリ 1 つ（中の `*.mcap` 全部）または `.mcap` ファイル 1 つが 1 走行になる。
- bag に /tf_static が無い場合は、キャリブレーションのファイルを `--tf-static` で渡す（形式は `configs/default.yaml`）。
- 毎回同じ指定をするなら設定ファイルにまとめる: `configs/example.yaml` をコピーして編集し、`./autolabel run my_area.yaml`。
  書いた項目だけが `configs/default.yaml` を上書きする。コマンドラインの指定はさらにそれを上書きする。
- **ログの追加**: `logs` の場所に新しいログを置いて `run` するだけでよい。
- **再開と再処理**: 終わった処理は飛ばし、入力・設定が変わったものと失敗したものだけをやり直す。
  1 本の bag やシーンが失敗しても、他は続行する（終了コード 1、`status` に失敗とログの場所）。
- **範囲を絞る**: `--drive` / `--scene` で対象を絞れる。`--force` で強制的にやり直す。
- **大量処理**: `--no-debug`（または設定の `debug: false`）でデバッグ出力を止め、`prune_frames: true` で容量を抑える。
  並列数は `jobs`。

## 出力（`output/<name>/`、`--output` / 設定 `output` で変更可）
```
dataset/                    学習データだけ（docs/DATASET.md）。このディレクトリだけを学習機に持っていけばよい
  dataset.json              入口: クラス・ベクトルクラス・既定の BEV・シーン一覧（train / val は場所で分割）・元 bag
  <scene>/map.npz           ラベル（地図座標のラスタ 0.1 m + 路面の高さ）
  <scene>/vectors.geojson   3D ベクトル（地図座標）
  <scene>/samples.jsonl     サンプル: 時刻・自車姿勢・各カメラの現在 / 過去フレーム・QA・元の bag と時刻
  <scene>/frames.jsonl, images/cam<N>/  入力画像（歪み補正済み）とその姿勢
debug/                      確認用（消してよい / 無効化できる）
  scenes/<scene>/           レビュー用シート（カメラ画像へのラベル重ね描き、地図、矢印一覧）、プレビュー、
                            PCD（LiDAR 路面点・ラベル・ベクトル）、LiDAR との幾何チェック
  logs/, dataset_stats.json, qa/, eval/
work/                       再開用の中間データ（走行ごとの画像・姿勢・セグメンテーション・キャリブ、シーンごとの路面・ラベル）
status.json                 処理単位ごとの成否・メモ・ログ
```
学習側の参照ローダ: `hdlabel/loader.py`。範囲と解像度はローダで自由に選べる（ラベルは地図座標で 1 回だけ持つ）。

## 仕組み（`hdlabel/steps/`）
| 処理 | 中身 |
|---|---|
| `ingest.py` | bag を 1 回読み、次のことをする。<br>・画像を 1/2 に縮小し、歪み補正して保存する。<br>・自己位置・車速・/tf_static を保存する。<br>・各 bag の時間範囲（元 bag の追跡用）を記録する。<br>・不正データを検出する（自己位置の欠落・跳び、カメラのコマ落ち、壊れた bag や画像は読める所まで使って記録）。<br>・キーフレーム（0.5 m ごと）を選ぶ。 |
| `segment.py` | キーフレームを Mask2Former（Mapillary Vistas）でセグメンテーションする。 |
| scenes（`run.py`） | 走行を、地図座標の固定の 300 m グリッドのマスごとにシーンに分ける。<br>・ラベル範囲は、その経路の前後 150 m と横 40 m（BEV の前方まで教師が入る）。<br>・同じ経路を走った別の走行のシーンは、ラベルを共有する（計算は 1 回）。 |
| `time_offset.py` | カメラ画像の時刻ずれを推定する。曲がっているときの白線のぶれが最小になるずれを探す。外部パラメータは公式値をそのまま使う。 |
| `map_surface.py` | 点群地図から自車の走る階層の路面 3D 形状を作る（`hdlabel/surface.py`）。<br>・点を高さの層に分ける。<br>・走行軌跡から段差の小さい層だけをたどって選ぶ。<br>・そのため、高架の下では地面、橋の上では橋面になり、別の階層は混ざらない。 |
| `project_labels.py` | 全走行・全カメラのセグメンテーションを光線として路面に交差させ（GPU）、0.1 m セルに投票する。<br>・融合するのは、同じ階層を走った走行だけ。<br>・別の階層にいた瞬間のフレームは使わない。 |
| `arrows.py` | 矢印ごとに、近い（2.5〜11 m）カメラ画像から 2 cm 解像度の鮮明な路面パッチを作る。<br>・実物から切り出した型（直進・右折・直進+左折）と照合し、向きと進行可能方向を決める（`hdlabel/arrowshape.py`）。<br>・ラスタは、測った輪郭で描き直す。 |
| `vectorize.py` | ラベル地図から次のベクトルを作る。<br>・白線（実線 / 破線、塗装区間）<br>・横断歩道<br>・縁石<br>・道路境界（本物の境界か、観測範囲の端か）<br>・走行可能領域<br>・矢印<br>レビュー修正（`review/<area>/*.json`、地図座標）を適用する。 |
| `export.py` | 学習データを書き出す。サンプルは基準カメラの時刻で、train は 0.5 s、val は 0.1 s 間隔（毎周期の評価用）。<br>QA で次の場合を不合格にする。<br>・近距離が未観測<br>・時刻ずれの推定失敗<br>・自己位置の欠落<br>・レビューで却下 |
| `review.py`, `geom_check.py` | デバッグ出力を作る（`docs/ANNOTATION.md`）。<br>・カメラ画像にラベルを重ねたシート、地図、矢印一覧<br>・LiDAR の段差と縁石の位置の照合、ベクトルの高さと LiDAR 地面の照合（カメラと独立な幾何チェック）<br>・PCD |

| 文書 | 内容 |
|---|---|
| [docs/DATASET.md](docs/DATASET.md) | 学習データの形式（座標系・ラスタ・ベクトル・複数フレーム入力・ローダ） |
| [docs/ANNOTATION.md](docs/ANNOTATION.md) | 目視レビューと修正ファイルの書き方 |
| [docs/MAINTENANCE.md](docs/MAINTENANCE.md) | コード構成、テスト・回帰確認、既知の制約、落とし穴 |
| [env/README.md](env/README.md) | Python 環境の再現 |

設定はすべて `configs/default.yaml`（各項目にコメントあり）。特定の場所のパス・キャリブレーション・レビュー結果は
`configs/local/` と `review/` に置けば git に入らない。

## ライセンス
Apache License 2.0（[LICENSE](LICENSE)）。

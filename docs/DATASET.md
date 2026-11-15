# 学習データの形式（format 2）

想定するモデル: 複数カメラ × 複数フレームを入力に、毎周期、自車まわり（200〜300 m）の HD Map を出力する
（ラスタ = BEV セグメンテーション、またはベクトル = MapTR 系）。両方の教師を同じラベルから出す。

**ラベルはシーンごとに地図座標で 1 回だけ持ち、サンプルは姿勢だけを持つ**（nuScenes の地図と同じ考え方）。
- 範囲・解像度・向きは、学習時にローダで決められる。データを作り直す必要はない。
- 細かい形状は解像度の制約を受けない。白線・矢印の輪郭はベクトルで cm 単位、ラスタは 0.1 m で持つ。
- 1 サンプルずつ BEV を保存しないので、容量が小さい。
- 各シーンのラベル範囲は、サンプルを取る経路（300 m グリッドの 1 マス内）の前後 150 m ＋ 横 40 m。前方 150 m まで教師が入る。

## ディレクトリ（`output/<name>/dataset/`）
```
dataset.json                 入口（下記）
<scene>/
  scene.json                 カメラ（K, size, T_base_cam, 時刻ずれとその確かさ）、既定の BEV、元の bag、履歴の取り方
  map.npz                    cls (uint8, 0.1 m) / height (float32, 0.2 m 路面高さ, NaN = 路面なし) / lo / res / surf_res / origin
  vectors.geojson            3D ベクトル（地図座標 m、z = 路面の高さ）
  samples.jsonl              1 行 = 1 サンプル
  frames.jsonl               1 行 = 1 入力フレーム（サンプルが使う画像だけ）
  images/cam<N>/<stamp>.jpg  歪み補正済み画像（work/ へのハードリンク。容量は増えない。コピーすると実体になる）
```
パスはすべてシーンのディレクトリからの相対パス。`dataset/` ごと別のマシンに持っていける。

## 座標系
- `map`: Autoware の地図座標（m）。点群地図と同じ。
- `base_link`: 自車（x 前、y 左、z 上）。**z は地図路面に接地させた値**（自己位置の z は路面より 10〜15 cm 低いので補正済み）。
- カメラ: OpenCV 系（x 右、y 下、z 前）。`T_base_cam` でカメラから base_link へ移す。
- 画像: 歪み補正済みのピンホール（`K`, `size`）、元画像の 1/2 解像度。
- 時刻: ns。`stamp` は画像ヘッダの時刻。`pose_stamp = stamp + time_offset_s` は、推定した時刻ずれを足した、姿勢を引く時刻。

## dataset.json
`format`, `name`, `cameras`（カメラ id）, `classes`（ラスタのクラス名）, `vector_classes`, `arrow_directions`, `bev`（既定の切り出し: `x`, `y` [m], `res`）,
`split`（`tile_m`, `val_every`, サンプル数）, `scenes`（`scene`, `drive`, `path`, `samples`, `qa_ok`, `split`, `tile`, `bags`, `t_range`）。

## samples.jsonl（1 行）
```json
{"stamp": 1793863477914273793,                 // サンプル時刻 (ns, 姿勢の時刻)
 "T_map_base": [[...4x4...]],                  // サンプル時刻の自車姿勢（接地補正済み）
 "frames": {"8": [120, 98, 77, 56, 35], ...},  // カメラごとの frames.jsonl の id: [現在, 0.5 s 前, 1.0 s 前, ...]
 "dt_ms": {"8": 0.0, "5": -24.5, ...},         // 現在フレームの姿勢時刻 - サンプル時刻
 "qa": {"observed_near": 0.93, "max_dt_ms": 48.0, "pose_ok": true, "calib_ok": true, "ok": true},
 "source": {"drive": "20261105-162434", "bag": "<元の bag ファイル名>", "stamp": 1793863477914273793}}
```
- 学習には `qa.ok == true` のサンプルだけを使う。不合格になるのは次の場合。
  - 近距離（前方 30 m・左右 8 m）の観測率が 50% 未満
  - 現在フレームの時刻差が 60 ms を超える
  - 自己位置が欠落している、または跳んでいる（ingest で検出）
  - そのドライブのカメラに時刻ずれ推定の失敗がある
  - レビューで却下された
- 履歴の間隔と枚数は `scene.json` の `history`（既定 0.5 s × 4 枚 = 2 s）に入っている。
- `source` から元の rosbag ファイルと時刻を辿れる。各フレームにも `bag` がある。

## frames.jsonl（1 行）
`{"id": 120, "cam": 8, "image": "images/cam8/<stamp>.jpg", "stamp": ..., "pose_stamp": ..., "T_map_base": [[...]], "bag": "..."}`
- `T_map_base` は、そのフレームの姿勢時刻での自車姿勢。
- 過去フレームの特徴を現在の自車座標に重ねるには `inv(T_map_base[サンプル]) @ T_map_base[フレーム] @ T_base_cam` を使う
  （ローダの `T_cur_frame`）。

## ラスタ（map.npz → 自車座標の BEV）
`hdlabel/bev.py` の `SceneMap.crop(T_map_base, x, y, res)` で切り出す。
- 格子の向き: 行 0 = 前方（x 最大）、列 0 = 左（y 最大）。
- 各セルの値: base_link の z 軸に沿って 3D 路面と交わる点のクラスを取る。坂でもずれない。
- `height`: その点の base_link での z。

| id | クラス | 説明 |
|---|---|---|
| 0 | unobserved | どのカメラからも見えていない / 路面外（**損失から除外**） |
| 1 | road | 車道 |
| 2 | marking | 白線・停止線・矢印など（横断歩道以外の路面標示）。矢印は鮮明な単フレームから測った輪郭で描き直してある |
| 3 | crosswalk | 横断歩道 |
| 4 | manhole | マンホール |
| 5 | curb | 縁石 |
| 6 | sidewalk | 歩道 |
| 7 | offroad | 植生・地面など車道外 |
| 8 | other | （路面上では出ない） |

`map.npz` の `cls[i, j]` は地図座標 `x = origin.x + lo.x + (i + 0.5) * res`、`y = origin.y + lo.y + (j + 0.5) * res`。
**ラスタに入るのは自車と同じ階層の路面だけ**。高架の下を走るときは地面の道路、橋の上では橋面になり、別の階層は混ざらない。

## ベクトル（vectors.geojson → 自車座標）
`SceneMap.vectors(T_map_base, x, y)` で base_link 座標に変換し、範囲で切り出す。
- 線は範囲を出入りする所で分割する。多角形は範囲との共通部分を取る。
- 点は 5 cm の許容で間引くだけで、形状は落とさない。

| kind | 形 | 主な属性 |
|---|---|---|
| `lane_line` | `points` 線 | `style`: solid / dashed / short、`painted`（塗装の割合）、`segments`（塗装区間 [s0, s1] m、見えている部分の先頭から）、`length` |
| `arrow` | `points` 線（尾 → 先端） | `directions`: `straight` / `left` / `right` の組（例 `["straight", "left"]`）、`shape`: S / L / R / SL / SR、`outlines`（塗装の輪郭リング、3D）。右折専用レーンなどの判定用 |
| `curb` | `points` 線 | |
| `road_boundary` | `points` 線 | `observed`: true = 本物の境界（外側が縁石・歩道・路外）、false = 観測範囲の端（**境界として学習しない**） |
| `crosswalk` | `rings` 多角形 | `type`（レビューで付く、例 `red_white`） |
| `marking_area` | `rings` 多角形 | 太い停止線・文字・ゼブラ帯など。`type`（レビューで付く: zebra / stop_line / text / …） |
| `drivable_area` | `rings` 多角形（穴あり） | |

MapTR 系との対応（ローダの `VECTOR_CLASSES`）は次の通り。
- divider = `lane_line`
- boundary = `curb` + `road_boundary`（observed）
- ped_crossing = `crosswalk`
- arrow = `arrow`（点列の向き = 矢印の向き、`vec_dir` = 進行可能方向）

## 参照ローダ（`hdlabel/loader.py`）
```python
from hdlabel.loader import HDMapDataset
ds = HDMapDataset("output/<name>/dataset/dataset.json", split="train",      # cams=[...] で使うカメラと順番を選べる（既定: 全カメラ）
                  history=3, image_size=(704, 448), bev=dict(x=[-50, 150], y=[-80, 80], res=0.2))
x = ds[0]   # images [T, C, 3, H, W], cam_mask, K, T_base_cam, T_cur_frame [T, C, 4, 4], frame_dt,
            # bev_cls / bev_height [nx, ny], vec_pts [V, 20, 3] / vec_cls / vec_mask / vec_dir [V, 3]
els = ds.item_vectors(0)   # 全属性つきのベクトル（白線の塗装区間、矢印の輪郭、多角形）
```
- 走行によって欠けているカメラがある（bag に無いカメラ）。欠けたカメラは `cam_mask` が false で、画像は 0。
- `bev` を省くと `dataset.json` の既定値で切り出す。

## train / val の分け方
同じ場所を何度も通るので、ランダムに分けると検証が甘くなる。そこで**場所で分けている**。
- 地図を `split.tile_m`（既定 250 m）のタイルに区切り、5 つに 1 つ（固定のハッシュ）を val にする。
- シーンの経路の中心がどのタイルに入るかで、そのシーンの split が決まる。
- val シーンはモデルの周期（`val_every_s`、0.1 s）でサンプルを出している。時刻順に再生して毎周期評価できる
  （`hdlabel/eval/online_eval.py`）。

## 学習できることの確認（`hdlabel/eval/train_smoke.py`、モデル開発ではない）
小さな BEV ネットで、次のことを確かめる。
- 画像・キャリブ・姿勢・ラベルの対応が正しいこと
- 教師がカメラ入力から学習できる形式であること

ネットの構成: 平面仮定で ResNet18 の特徴を BEV セルの投影位置でサンプリングし、0.8 m セルで学習する。
val を「各セルで train に最も多いクラス」と比べる。結果は `output/<name>/debug/eval/` に出る。

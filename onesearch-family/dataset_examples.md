# SFT Stage 1: Semantic Alignment

## V1

### 1. query ↔ sid

| Type | Input | Output |
|------|------|------|
| query → sid | 用户query为`<苹果十七迷你>` | id为`<a_3558><b_477><c_161><d_191><d_324>` |
| sid → query | id为`<a_2831><b_996><c_186><d_104><d_368>` | 对应的用户query为`<贝诺酯片100片>` |

### 2. item ↔ sid

| Type | Input | Output |
|------|------|------|
| item → sid | 商品标题为`<【利灵】Kitty生活鱼快字母钥匙扣亚克力高颜值挂饰礼物挂件>` | id为`<a_1806><b_200><c_403><d_52><d_313>` |
| sid → item | id为`<a_2538><b_452><c_326><d_53><d_431>` | 对应的商品标题为`<便携式卡式炉气罐户外卡磁炉丁烷液化气燃气瓦斯气体小气瓶煤气罐>` |

### 3. query/item → cate

| Type | Input | Output |
|----------|------|----------|
| item | 商品标题为`<朵莱人宠物家庭专用消毒泡腾片环境消毒空气消毒不伤手快速泡腾>` | `宠物生活/宠物消毒剂` |
| query | 用户query为`<轻松熊的学习东西>` | `文化用品/中性笔` |

### 4. sid → cate

| Input | Output |
|------|----------|
| id为`<a_1382><b_303><c_48><d_181><d_470>` | `女装/休闲裤` |

---

## V2 + CoT Tasks

### 1. query → keywords

| Input | Output |
|------|------|
| 用户当前搜索query为`<cleanfit>` | `束脚运动裤,透气速干外套,中性风休闲鞋,多功能健身包` |

### 2. query_sid → keyword_sid

| Input | Output |
|------|------|
| 用户当前搜索query的id为`<a_89><b_905><c_445><d_223><d_489>` | `<a_3854><b_386><c_147><d_37><d_328>` |

### 3. query + [user_info] → P_keywords（personalized keywords）

| Input | Output |
|------|------|
| 用户当前搜索query为`<#cleanfit>`，用户为`18-23岁男性`，id为`{short_sid + long_sid}`，最近搜索query有`<哈灵顿夹克><cleanfit竹节蓝><消防斧><vivox200pro><国潮穿搭>`，历史交互商品id有`<a_498><b_338><c_46><d_53><d_299>`和`<a_3109><b_414><c_142><d_204><d_378>`和`<a_3109><b_522><c_68><d_126><d_388>`和`<a_3109><b_114><c_67><d_126><d_388>`和`<a_641><b_400><c_140><d_55><d_403>`，请给出1-5个关键词：| `cleanfit穿搭,工装牛仔裤,美式短袖T恤,微喇牛仔裤男,冰蓝牛仔裤` |


### 4. query_sid + [user_info] → P_keyword_sid (personalized keyword sid)
| Input | Output |
|------|------|
| 用户当前搜索query的id为`<a_97><b_658><c_209><d_2><d_466>`，用户为`18-23岁男性`，id为`{short_sid + long_sid}`，最近搜索query有`<哈灵顿夹克><cleanfit竹节蓝><消防斧><vivox200pro><国潮穿搭>`，历史交互商品id有`<a_498><b_338><c_46><d_53><d_299>`和`<a_3109><b_414><c_142><d_204><d_378>`和`<a_3109><b_522><c_68><d_126><d_388>`和`<a_3109><b_114><c_67><d_126><d_388>`和`<a_641><b_400><c_140><d_55><d_403>`，请给出合适的关键词id： | `<a_97><b_658><c_209><d_2><d_467>` |

---

# SFT Stage 2: Q-I Co-occurrence

### 1. query ↔ item

| Type | Input | Output |
|------|------|------|
| query → item | 用户当前搜索query为`<李欣蕊过桥米线>` | 商品标题：`【双十一囤货自由】方便速食李梓睿过桥米线360g*10袋` |
| item → query | 用户当前点击的商品标题为`<小米/红米 Redmi Note 15 Pro手机学生智能正品游戏9新便宜手机>` | 搜索query：`红米骁龙8s快充大电量` |

### 2. query_sid ↔ item_sid

| Type | Input | Output |
|------|------|------|
| query_sid → item_sid | 用户当前搜索query的id为`<a_89><b_905><c_445><d_223><d_489>` | 商品id：`<a_3854><b_386><c_147><d_37><d_328>` |
| item_sid → query_sid | 用户当前点击商品的id为`<a_2365><b_849><c_259><d_130><d_371>` | 搜索query的id：`<a_3580><b_636><c_138><d_50><d_374>` |

---

# SFT Stage 3: User Personalization

### query + query_sid + [user_info] → item_sid

| Field | Content |
|------|------|
| query | `<东诗品质女装>` |
| query_sid | `<a_2421><b_132><c_52><d_90><d_462>` |
| User_profile | `50+ 岁女性` |
| User_id (sensitive) | `{short_sid + long_sid}` |
| Lastest search queries (only partially shown) | `<小衫>` `<太阳能户外灯庭院灯>` `<时尚小衫弹力>` `<时尚小衫>` `<时尚套装洋气时髦时尚气质>` |
| Historial interacted item sids (short, only partially shown) | `<a_3571><b_162><c_402><d_126><d_388>` |
| Historial interacted item embeddings (long, inserted in model) | `{three token embeddings}` |
| **Output** | `<a_163><b_788><c_247><d_34><d_317>` |


| Input | Output |
|------|------|
| 用户当前搜索query的id为`<a_97><b_658><c_209><d_2><d_466>`，用户为`18-23岁男性`，id为`{short_sid + long_sid}`，最近搜索query有`<哈灵顿夹克><cleanfit竹节蓝><消防斧><vivox200pro><国潮穿搭>`，历史交互商品id有`<a_498><b_338><c_46><d_53><d_299>`和`<a_3109><b_414><c_142><d_204><d_378>`和`<a_3109><b_522><c_68><d_126><d_388>`和`<a_3109><b_114><c_67><d_126><d_388>`和`<a_641><b_400><c_140><d_55><d_403>`和`{three token embeddings}`，请给出合适的商品id： | `<a_97><b_658><c_209><d_2><d_467>` |

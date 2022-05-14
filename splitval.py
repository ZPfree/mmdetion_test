from mmdet.datasets.api_wrappers import COCO
import os,json
import random
root='K:/FEWYANGBEN/train/'
coco = COCO(os.path.join(root, 'annotations/instances_train2017.json'))
class_names = [coco.cats[catId]['name'] for catId in coco.getCatIds()]
# for cname in class_names:
#     print(cname)#我是看获取的类别名字是否有乱码

categories = [dict(id=i+1, name=name) for i, name in enumerate(class_names)]

annotaions_train = []
images_train = []
annotaions_val = []
images_val = [] #getImgIds
# for images_id in coco.getImgIds():
# #getImgIds
#     print("收集到图片id",images_id)
#按照图片类别去划分val,test,每个类别的数据抽取百分之十做验证

for catId in coco.getCatIds():
    print("收集的catID",catId)
    imgIds = coco.getImgIds(catIds=[catId])
    random.shuffle(imgIds)
    splitv = int(len(imgIds)*0.1)
    print("类别验证集个数",splitv)
    for imgId in imgIds[:splitv]:
        img = coco.imgs[imgId]
        images_val.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_val.append(ann)
    for imgId in imgIds[splitv:]:
        img = coco.imgs[imgId]
        images_train.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_train.append(ann)
json_dict_train = {"images": images_train, "type": "instances", "annotations": annotaions_train, "categories": categories}
json_dict_val = {"images": images_val, "type": "instances", "annotations": annotaions_val, "categories": categories}
with open('K:/FEWYANGBEN/val/val.json', "w",encoding='utf-8') as f:
    json.dump(json_dict_val, f ,ensure_ascii=False)#ensure_ascii=False 确保中文不乱码
    print("traon加载文件完成")
with open('K:/FEWYANGBEN/val/test.json', "w",encoding='utf-8') as f:
    json.dump(json_dict_train, f ,ensure_ascii=False)
    print("2加载文件完成")
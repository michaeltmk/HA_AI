from object_detection.utils import label_map_util


class GetNumClasses:
    @classmethod
    def get_num_classes(pbtxt_fname):
        
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())
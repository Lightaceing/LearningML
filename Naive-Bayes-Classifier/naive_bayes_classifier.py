from helper import find_categories, read_json, write_catgory_file, get_test_data


# reading train docs
train_docs = read_json("train_file.json")

# reading test docs
test_text = get_test_data("test.json")

# category name list
category_name_list = ['+1', '-1']

# predicting category
predicted_list = find_categories(train_docs=train_docs, test_docs=test_text,
                                 category_names=category_name_list)

# writing predicted category in another file
write_catgory_file(predicted_list)

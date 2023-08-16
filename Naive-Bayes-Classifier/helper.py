import json


def calc_class_prob(doc, category_name):
    """
    Input : doc : a dict with all text and corresponding category
           category_name : a list with all category names

    Return : list with all prob, ordered as i/p arg 2

    Calc categorical prob. as naive bayes classifier
    """
    total_sets = len(doc)

    # creating a dict with all 0s
    count = []
    for i in category_name:
        count.append(0)
    temp_dict = dict(zip(category_name, count))

    # adding count
    for text in doc:
        category = doc[text]
        temp_dict[category] += 1

    # calc prob of each category and saving in list
    prob_categories = []
    for keys in temp_dict:
        prob = temp_dict[keys]/total_sets
        prob_categories.append(prob)

    return prob_categories


def freq_count_class(doc, category):
    """
    Input : doc : a dict with all text and corresponding category
           category : category to be counted from

    Return : list with all prob, ordered as i/p arg 2
    Provides Freq. count for each 
    """
    temp_dict = {}

    for text in doc:
        if (doc[text] == category):
            temp_list = text.split(' ')
            for item in temp_list:
                if (item not in temp_dict):
                    temp_dict.update({item: 1})
                else:
                    temp_dict.update({item: temp_dict[item]+1})
    return temp_dict


def get_vocab(doc):
    vocab_list = []
    for text in doc:
        for item in text.split(' '):
            if item not in vocab_list:
                vocab_list.append(item)
    return vocab_list


def remove_unique(text, doc):
    vocab_list = get_vocab(doc)
    string = ''
    temp_list = text.split(' ')
    for item in temp_list:
        if (item in vocab_list):
            string = string + ' ' + item
    return string.strip()


def count_word_in_each_class(cleaned_word, vocab_list):

    cleaned_word_list = cleaned_word.split(' ')
    freq_word = []

    for each_dict in vocab_list:
        temp_dict = {}

        for each_item in cleaned_word_list:
            if (each_item in each_dict.keys()):
                temp_dict.update({each_item: each_dict[each_item]})
            else:
                temp_dict.update({each_item: 0})

        freq_word.append(temp_dict)

    return freq_word


def calc_naive_prob(docs, vocab_list, freq_word, category_count):

    total_vocab = len((get_vocab(docs)))

    prob_list = []  # list of each prob sep.
    temp_list = []
    # list having total class count
    for item in vocab_list:
        temp_list.append(len(get_vocab(item))+1)

    # calc of each word

    for item, count in zip(freq_word, temp_list):  # first pos then neg
        garbage_list = []
        for key in item:

            num = (item[key]+1)  # +1 for smoothening
            den = (total_vocab+count)
            final_val = num/den
            garbage_list.append(final_val)
        prob_list.append(garbage_list)

    del temp_list, garbage_list

    val_list = []
    for val in prob_list:
        value = 1
        for each_item in val:
            value = value*each_item
        val_list.append(value)

    # memory clean-up
    del prob_list

    max_val = max(val_list)

    final_list = []
    for each_item, each_value in zip(category_count, val_list):
        final_list.append(each_item*each_value)

    # finding index to find category
    index = val_list.index(max_val)

    return index


def make_vocab_list(docs, category_names):
    cat_count = []
    for name in category_names:
        cat_count.append(freq_count_class(docs, name))
    return cat_count


def find_category_on_one(train_docs, test_docs, category_names):

    category_count = calc_class_prob(train_docs, category_names)
    vocab_list = make_vocab_list(train_docs, category_names)
    cleaned_word = remove_unique(test_docs, train_docs)
    freq_word = count_word_in_each_class(cleaned_word, vocab_list)
    index = calc_naive_prob(train_docs, vocab_list, freq_word, category_count)

    return category_names[index]


def find_categories(train_docs, test_docs, category_names):
    temp_dict = {}
    for each_doc in test_docs:
        category = find_category_on_one(
            train_docs=train_docs, test_docs=each_doc, category_names=category_names)

        temp_dict.update({each_doc: category})

    return temp_dict


def write_catgory_file(dicto):
    with open("data_file.json", "w") as write_file:
        json.dump(dicto, write_file)
    print('Category classified.\nFile created.')


def get_test_data(file_name):
    cri = []
    test_text = read_json(file_name)
    for item in test_text:
        for each_item in test_text[item]:
            cri.append(each_item)
    return cri


def read_json(file_name):
    file_name = file_name
    f = open(file_name,)
    data = json.load(f)
    return data

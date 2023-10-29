
def Enhance_wmape(list1,list2):#list1为真实值
    diff_sum = 0
    for i in range(len(list1)):
        diff_sum += abs(list1[i] - list2[i])

    sum_list1 = sum(list1)
    wmape=(1-(diff_sum/sum_list1))*100

    formatted_wmape = "{:.2f}%".format(wmape)

    return formatted_wmape




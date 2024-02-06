import random

#Monte_Carlo随机模拟检验检验是否为马尔可夫链

#网球规则更新函数
def clu_games_numbers(player_1_points_numbers, player_2_points_numbers, player_1_games_numbers, player_2_games_numbers,player_1_sets_numbers,player_2_sets_numbers):
    game=1
    if not (player_1_games_numbers==6 and player_2_games_numbers==6):
        if player_1_points_numbers == 4 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_1_points_numbers == 5 and player_2_points_numbers == 3:
            player_1_games_numbers += 1
        elif player_1_points_numbers > 4 and player_2_points_numbers > 4 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_2_points_numbers == 4 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        elif player_2_points_numbers == 5 and player_1_points_numbers == 3:
            player_2_games_numbers += 1
        elif player_2_points_numbers > 4 and player_1_points_numbers > 4 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        else:
            game=0
    elif player_1_sets_numbers+player_2_sets_numbers<=4:
        if player_1_points_numbers == 7 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_1_points_numbers == 8 and player_2_points_numbers == 6:
            player_1_games_numbers += 1
        elif player_1_points_numbers > 7 and player_2_points_numbers > 7 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_2_points_numbers == 7 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        elif player_2_points_numbers == 8 and player_1_points_numbers == 6:
            player_2_games_numbers += 1
        elif player_2_points_numbers > 7 and player_1_points_numbers > 7 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        else:
            game=0
    elif player_1_sets_numbers+player_2_sets_numbers==5:
        if player_1_points_numbers == 10 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_1_points_numbers == 11 and player_2_points_numbers == 9:
            player_1_games_numbers += 1
        elif player_1_points_numbers > 10 and player_2_points_numbers > 10 and player_1_points_numbers - player_2_points_numbers >= 2:
            player_1_games_numbers += 1
        elif player_2_points_numbers == 10 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        elif player_2_points_numbers == 11 and player_1_points_numbers == 9:
            player_2_games_numbers += 1
        elif player_2_points_numbers > 10 and player_1_points_numbers > 10 and player_2_points_numbers - player_1_points_numbers >= 2:
            player_2_games_numbers += 1
        else:
            game=0
    else:
        game=0
    return player_1_games_numbers, player_2_games_numbers,game


def clu_sets_numbers(player_1_games_numbers, player_2_games_numbers, player_1_sets_numbers, player_2_sets_numbers):
    set=1
    if player_1_games_numbers == 6 and player_1_games_numbers - player_2_games_numbers >= 2:
        player_1_sets_numbers += 1
    elif (player_1_games_numbers == 7 and player_2_games_numbers == 5) or (player_1_games_numbers == 7 and player_2_games_numbers == 6):
        player_1_sets_numbers += 1
    elif player_1_games_numbers > 6 and player_2_games_numbers > 6 and player_1_games_numbers - player_2_games_numbers >= 2:
        player_1_sets_numbers += 1
    elif player_2_games_numbers == 6 and player_2_games_numbers - player_1_games_numbers >= 2:
        player_2_sets_numbers += 1
    elif (player_2_games_numbers == 7 and player_1_games_numbers == 5) or (player_2_games_numbers == 7 and player_1_games_numbers == 6):
        player_2_sets_numbers += 1
    else:
        set=0
    return player_1_sets_numbers, player_2_sets_numbers,set

def Monte_Carlo(abi,pro):
    pass


def random_zero_one():
    a=random.choice([0, 1])
    if a==0:
        return 0,1
    else:
        return 1,0

for i in range(10):
    player_1_games_numbers=0
    player_2_games_numbers=0
    player_1_sets_numbers=0
    player_2_sets_numbers=0
    player_1_points_numbers=0
    player_2_points_numbers=0
    for j in range(1000):
        this_player_1_points_numbers,this_player_2_points_numbers = random_zero_one()#每次返回0、1或1、0
        player_1_points_numbers+=this_player_1_points_numbers
        player_2_points_numbers+=this_player_2_points_numbers
        player_1_games_numbers,player_2_games_numbers,game = clu_games_numbers(player_1_points_numbers, player_2_points_numbers, player_1_games_numbers, player_2_games_numbers,player_1_sets_numbers,player_2_sets_numbers)
        if game==1:#说明局数发生了变化
            player_1_points_numbers=0
            player_2_points_numbers=0
        player_1_sets_numbers,player_2_sets_numbers,set = clu_sets_numbers(player_1_games_numbers,player_2_games_numbers, player_1_sets_numbers, player_2_sets_numbers)
        if set==1:#说明盘数发生了变化
            player_1_points_numbers=0
            player_2_points_numbers=0
            player_1_games_numbers=0
            player_2_games_numbers=0
        #打印比赛结果
        print('now is ','The Set:',player_1_sets_numbers+player_2_sets_numbers,'The Game:',player_1_games_numbers+player_2_games_numbers,'The Point:',player_1_points_numbers+player_2_points_numbers)
        print('after this point ,the points games and sets of player_1 is: ',player_1_points_numbers,player_1_games_numbers,player_1_sets_numbers)
        print('after this point ,the points games and sets of player_2 is: ', player_2_points_numbers,player_2_games_numbers,player_2_sets_numbers)
        if player_1_sets_numbers==3 or player_2_sets_numbers==3:
            print('this range is over')
            break



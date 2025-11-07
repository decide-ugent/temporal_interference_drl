
# function to get first and last oven checks
def get_oven_checks(actions, oven_timer, target_duration):
    first_oven_check = []
    last_oven_check = []
    i = 0
    for i in range(len(actions)):
        # Start: action 5 leads to oven turning on
        if actions[i] == 5 and oven_timer[i] >= target_duration:  # action oven check and oven opens
            # print(i)
            j = i
            while actions[j] == 5:  # go back until action is 5 consequently
                j = j - 1
                continue
            first_oven_check.append(oven_timer[j + 1])
            last_oven_check.append(oven_timer[i])
            # print(j+1)

    return first_oven_check, last_oven_check
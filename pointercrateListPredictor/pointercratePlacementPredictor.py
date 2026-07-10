import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import calendar as cld
import random
import json

numMonthlyPlacements = {}
levelPlacementFrequencyList = {}
stabilityFreq = {}
originalLevelRanks = {} #rank, levelName
levelRanks = {} #rank, levelName
levelSwapFreq = {}

def loadListData():
    return pd.read_csv("data/pointercrateListData.csv")

def findNumMonthlyPlacements(df):
    cols = df.columns
    for i in range(1, len(cols) - 1):
        current_col = cols[i]
        next_col = cols[i+1]

        noNewLevels = 0
        nextNoNewLevels = 0

        for index, row in df.iterrows():
            if row[current_col] == "NP":
                noNewLevels += 1

            if row[next_col] == "NP":
                nextNoNewLevels += 1

        numMonthlyPlacements[(nextNoNewLevels - noNewLevels)] = (numMonthlyPlacements.get((nextNoNewLevels - noNewLevels), 0) + 1)

    return numMonthlyPlacements

def findLevelPlacementsFreq(df):
    cols = df.columns
    for i in range(1, len(cols) - 1):
        current_col = cols[i]
        next_col = cols[i + 1]

        for index, row in df.iterrows():
            if row[current_col] != "NP" and row[current_col] != "LL" and row[next_col] == "NP":
                placement = int(row[current_col])
                levelPlacementFrequencyList[placement] = (levelPlacementFrequencyList.get(placement, 0) + 1)

    return levelPlacementFrequencyList

def findLevelStability(df): #doesn't work as intended

    cols = df.columns
    max_level = 150
    group_size = 10

    for i in range(len(cols) - 2, 0, -1):
        current_col = cols[i]
        next_col = cols[i + 1]

        for start in range(1, max_level + 1, group_size):
            end = start + group_size - 1
            list_diff = {}

            for r in range(len(df)):
                row = df.iloc[r]
                curr = row[current_col]
                nxt = row[next_col]

                if curr in ["LL", "NP"] or nxt in ["LL", "NP"]:
                    continue

                rank = int(curr)

                if start <= rank <= end:
                    movement = abs(int(curr) - int(nxt))
                    list_diff[row.iloc[0]] = movement

            if len(list_diff) > 0:
                diff = np.array(list(list_diff.values()))
                q1 = np.percentile(diff, 25)
                q3 = np.percentile(diff, 75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                for level, movement in list_diff.items():
                    row_index = df[df.iloc[:, 0] == level].index[0]

                    if df.iloc[row_index, 1] == "LL":
                        stabilityFreq.pop(level, None)
                        continue

                    if lower_bound <= movement <= upper_bound:
                        stabilityFreq[level] = stabilityFreq.get(level, 0) + 1
                    else:
                        levelSwapFreq[movement] = levelSwapFreq.get(movement, 0) + 1
                        stabilityFreq[level] = 0

    return stabilityFreq

def runSimulation():
    df = loadListData()

    for i in range(1, 151):
        # the dictionary becomes inaccurate by doing this, but for the purposes of the project this is fine
        levelPlacementFrequencyList[i] = levelPlacementFrequencyList.get(i, 0) + 1

    for index, row in df.iterrows():
        level = row.iloc[0]
        rank = row.iloc[1]
        if rank.isdigit():
            levelRanks[int(rank)] = level

    originalLevelRanks = levelRanks.copy()
    monthCount = 0
    newLevelCount = 1
    currentCheckpoint = dt.date.today()

    dataHistory = []
    dataHistory.append({"Date": currentCheckpoint.isoformat(), "List": originalLevelRanks})

    while bool(set(originalLevelRanks.values()) & set(levelRanks.values())):
        monthlyAdd = []
        placementAdd = []
        stabilityAdd = set()
        swapArr = []

        daysInMonth = cld.monthrange(currentCheckpoint.year, currentCheckpoint.month)[1]
        remainingDays = daysInMonth - currentCheckpoint.day + 1
        scale = remainingDays / daysInMonth
        nextCheckpointDate = currentCheckpoint + dt.timedelta(days=remainingDays)

        for key in numMonthlyPlacements.keys():
            for i in range(0, numMonthlyPlacements[key]):
                monthlyAdd.append(key)
        numOfNewLvls = round(random.choice(monthlyAdd) * scale)
        numMonthlyPlacements[numOfNewLvls] = numMonthlyPlacements.get(numOfNewLvls, 0) + 1

        for key in levelPlacementFrequencyList.keys():
            for i in range(0, levelPlacementFrequencyList[key]):
                placementAdd.append(key)
            random.shuffle(placementAdd)
        while len(placementAdd) > numOfNewLvls and len(placementAdd) > 0:
            placementAdd.pop(random.randint(0, len(placementAdd) - 1))
        for placement in placementAdd:
            levelPlacementFrequencyList[placement] = levelPlacementFrequencyList.get(placement, 0) + 1
            for rank in range(max(levelRanks.keys()), placement - 1, -1):
                levelRanks[rank + 1] = levelRanks[rank]
            levelRanks[placement] = "NL" + str(newLevelCount)
            newLevelCount = newLevelCount + 1

        for key in stabilityFreq.keys():
            for i in range(stabilityFreq[key]):
                stabilityAdd.add(key)
        prob = (1 / ((len(df.columns) - 2) + monthCount)) * scale
        for level in stabilityAdd.copy():
            if random.random() > prob:
                stabilityAdd.remove(level)

        for key in levelSwapFreq.keys():
            for value in range(levelSwapFreq[key]):
                swapArr.append(key)
        random.shuffle(swapArr)
        while len(swapArr) > len(stabilityAdd):
            swapArr.pop(random.randint(0, (len(swapArr)) - 1))
        for i in range(0, len(swapArr)):
            if random.random() < 0.5:
                swapArr[i] = -(swapArr[i])

        if len(swapArr) > 0:
            for i, movement in enumerate(swapArr):
                stabList = list(stabilityAdd)
                level = stabList[i]

                matching_ranks = [rank for rank, name in levelRanks.items() if name == level]

                if not matching_ranks:
                    continue

                current = matching_ranks[0]

                if current + movement < 1:  # done to ensure level ranks are positive, non-zero values
                    movement = -movement

                max_rank = max(levelRanks.keys())
                min_rank = min(levelRanks.keys())
                target = max(min(current + movement, max_rank), min_rank)

                if movement < 0:
                    while current > target:
                        levelRanks[current], levelRanks[current - 1] = (levelRanks[current - 1], levelRanks[current])
                        current -= 1
                elif movement > 0:
                    while current < target:
                        levelRanks[current], levelRanks[current + 1] = (levelRanks[current + 1], levelRanks[current])
                        current += 1
                elif movement == 0:
                    continue
                levelSwapFreq[abs(movement)] = levelSwapFreq.get(abs(movement), 0) + 1

        for level in stabilityAdd:
            if level in stabilityFreq:
                stabilityFreq[level] = 0

        for level in levelRanks.values():
            stabilityFreq[level] = stabilityFreq.get(level, 0) + 1

        for key in reversed(list(levelRanks)):
            if key > 150:  # LL handling
                level = levelRanks[key]
                del levelRanks[key]
                stabilityFreq.pop(level, None)

        monthCount += scale
        currentCheckpoint = nextCheckpointDate
        print("Months:", monthCount)
        print(currentCheckpoint,":", levelRanks,"\n")
        
        dataHistory.append({"Date": currentCheckpoint.isoformat(), "List": levelRanks.copy()})

        with open("data/predictedList.json", "w", encoding="utf-8") as file:
            json.dump(dataHistory, file, indent=4)

findNumMonthlyPlacements(loadListData())
findLevelPlacementsFreq(loadListData())
findLevelStability(loadListData())
print(runSimulation())
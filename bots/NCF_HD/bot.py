__author__ = '이다영, 박혜진'

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pathlib
import pickle
import time

from enum import Enum
import random
import math

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.data import Alert
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint
from sc2.position import Point2, Point3
from enum import Enum
from random import *

from .consts import CommandType, NukeStrategy

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
         #5가 state 개수, 12가 유닛종류(economy)
        self.fc1 = nn.Linear(5, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.vf = nn.Linear(64, 1)
        #self.combat_head = nn.Linear(64, len(CombatStrategy))
        self.nuke_head = nn.Linear(64, len(NukeStrategy))
        #self.mule_head = nn.Linear(64, len(MuleStrategy))

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        value = self.vf(x)
        #combat_logp = torch.log_softmax(self.combat_head(x), -1)
        nuke_logp = torch.log_softmax(self.nuke_head(x), -1)
        #mule_logp = torch.log_softmax(self.mule_head(x), -1)
        bz = x.shape[0]
        logp = (nuke_logp.view(bz, -1)).view(bz, -1)
        #logp = (combat_logp.view(bz, -1, 1) + nuke_logp.view(bz, 1, -1)).view(bz, -1)
        #logp = (product_logp.view(bz, 1, -1, -1) + nuke_logp.view(bz, -1, 1, -1) + mule_logp.view(bz, -1, -1, 1)).view(bz, -1)
        return value, logp

    
class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.pos=0
        self.dead=-1
        self.ghost_pos=0
        self.enemy_pos=0
        self.ghost_tag=0
        self.nuke_time = 0
        self.stop = False
        self.middle_alert = False

    def reset(self):
        self.ghost_pos = self.bot.start_location.x
        self.enemy_pos= self.bot.enemy_start_locations[0].x
        self.course = [Point2((self.ghost_pos,55)),Point2((self.enemy_pos,55)),Point2((self.ghost_pos,10)),Point2((self.enemy_pos,10))]
        if self.ghost_pos == 32.5:
            self.middle = 72.5
        else:
            self.middle = 52.5

    def runaway(self):
        pass # 아직 어떻게할지 모름

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        # 고스트 생산명령 내릴준비
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
        
        # 생산
        if ghosts.amount == 0 and self.bot.die_count <= 2:
            actions.append(cc.train(UnitTypeId.GHOST))
            

        if ghosts.amount > 0:

            if self.dead == 0 and self.bot.nuke_strategy == 0: # 위에서 죽었는데 또 위로가면
                self.bot.nuke_reward -= 0.01 #마이너스
            elif self.dead == 1 and self.bot.nuke_strategy == 1:
                self.bot.nuke_reward -= 0.01 #마이너스
            elif self.dead == 2 and self.bot.nuke_strategy == 2:
                self.bot.nuke_reward -= 0.01 #마이너스

            self.dead = 3 

            ghost = ghosts.first #고스트는 딱 한 개체만
            threaten = self.bot.known_enemy_units.closer_than(10, ghost.position)
            nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
            ravens = threaten(UnitTypeId.RAVEN)

            # 기지와 떨어졌을때 적 발견시 은폐
            if ghost.distance_to(self.bot.start_location) > 10 and threaten.amount > 0 :
                # 위 or 아래로 가고있는 와중, 고스트 혼자가 아닐땐 공격명령
                if not self.stop and nuke_units.amount > 1:
                    if ravens.amount > 0: # 밤까마귀이면 도망가는 코드 추가
                        for unit in nuke_units:
                            if unit.type_id == UnitTypeId.GHOST:
                                runaway()
                            else:
                                actions.append(unit.attack(ravens.closest_to(unit)))
                    else :
                        for unit in nuke_units:
                            # @@ 수정필요 상대팀이 우리보다 많으면 고스트는 은폐, 핵공격
                            if threaten.amount > nuke_units.amount:
                                actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                                actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.threaten.center))
                            if self.stop and unit.type_id == UnitTypeId.GHOST:
                                pass
                            else:
                                actions.append(unit.attack(threaten.closest_to(unit)))
                else : # 고스트 혼자이면 바로 은폐 쓰기
                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    if threaten.amount > 10 and self.bot.nuke_strategy == 2:
                        self.bot.ghost_ready = True
                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=threaten.center))

                
            # 위로
            if self.bot.nuke_strategy == 0 and self.pos != 3: 
                for unit in nuke_units:
                # 위치 이동
                    if unit.distance_to(self.course[0]) > 3 and self.pos == 0:
                        actions.append(unit.move(self.course[0]))
                        self.pos=1 # 올라가기

                    if unit.distance_to(self.course[0]) == 0 and self.pos == 1:
                        actions.append(unit.move(self.course[1]))
                        self.pos=2 # 옆으로 가기

                    if unit.distance_to(self.course[1]) < 3:
                        self.pos=3 # 대기장소 도착
            
            # 아래로 가라
            elif self.bot.nuke_strategy == 1 and self.pos != 3: 
                # 위치 이동
                for unit in nuke_units:
                    if unit.distance_to(self.course[2]) > 3 and self.pos == 0:
                        actions.append(unit.move(self.course[2]))
                        self.pos=1 # 올라가기

                    if unit.distance_to(self.course[2]) == 0 and self.pos == 1:
                        actions.append(unit.move(self.course[3]))
                        self.pos=2 # 옆으로 가기

                    if unit.distance_to(self.course[3]) < 3:
                        self.pos=3 # 대기장소 도착

            # 가운데로 가라 - 유령 혼자만 갈 거임
            elif self.bot.nuke_strategy == 2 and self.pos!=3:
                # 에너지가 50 이상일때, 가운데로 출발
                if ghost.energy > 70 and ghost.distance_to(Point2((self.middle,30))) > 3 and self.pos == 0:
                    actions.append(ghost.move(Point2((self.middle,30)))) #중간지점으로 가기
                    self.pos=1 

                # 가던도중 적을 만났거나, 무사히 가운데로 왔다면 은폐
                if ghost.distance_to(Point2((self.middle,30))) < 2 and self.pos == 1:
                    self.pos = 2

                if self.pos == 2 and ghost.is_idle:
                    
                    self.bot.ghost_ready = True
                    ghost_abilities = await self.bot.get_available_abilities(ghost)
                   
                    if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities: 
                        actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_start_locations[0]))
                    
                        

            if self.pos==3 or self.stop and ghost.is_idle:
                self.bot.ghost_ready = True # @@고스트 준비됐음 플레그, 핵 우선으로 생산할까??
                ghost_abilities = await self.bot.get_available_abilities(ghost)
                
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghost.is_idle:
                    # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_start_locations[0]))
                    self.nuke_time = self.bot.time
                    self.stop = True
                    self.bot.nuke_reward += 0.1


        elif ghosts.amount==0 and self.dead==3 : #고스트 죽음 (amount==0)
            self.pos=0
            self.bot.die_count += 1
            self.stop = False
            self.bot.nuke_reward -= 0.1

            if nuke_units.amount > 0:
                for unit in nuke_units:
                    actions.append(unit.move(self.bot.start_location))

            # @@핵 쏘는 도중 죽으면 마이너스?
            if self.bot.time - self.nuke_time < 14.0: 
                self.bot.nuke_reward -= 0.05

            if self.bot.nuke_strategy == 0: #윗길로 갔었으면
                self.dead = 0 #위에서죽음 표시
            elif self.bot.nuke_strategy == 1:
                self.dead = 1 #아랫길이었다면 아래서 죽음 표시
            elif self.bot.nuke_strategy == 2:
                self.dead = 2 #아랫길이었다면 아래서 죽음 표시
        
        return actions


class ReconManager(object):
    """
    정찰부대 운용을 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.a=0
        self.patrol_pos=list()
        self.front = None
        self.patrol_pos = None
        
        #self.p=queue.Queue()

    def patrol(self, center, combat_center):
        # 사령부만 정찰 혹은 combat 만을 정찰
        if combat_center == None:
            if center.x != self.bot.start_location.x : # 컴뱃정찰이면 좁게 정찰
                self.patrol_pos = [Point2((center.x, center.y+5)), Point2((center.x+5, center.y)), Point2((center.x, center.y-5)), Point2((center.x-5, center.y))]
            else:
                self.patrol_pos = [Point2((center.x, center.y+10)), Point2((center.x+10, center.y)), Point2((center.x, center.y-10)), Point2((center.x-10, center.y))]
        # 사령부와 deffense combat 을 정찰
        elif combat_center > center.x:
            self.patrol_pos = [Point2((center.x, center.y+10)), Point2((combat_center, center.y)), Point2((center.x, center.y-10)), Point2((center.x-10, center.y))]
        else:
            self.patrol_pos = [Point2((center.x, center.y+10)), Point2((combat_center, center.y)), Point2((center.x, center.y-10)), Point2((center.x+10, center.y))]
        
    def recon(self,raven,actions):
        if self.bot.is_ghost == 0:
            if self.a==0 :
                actions.append(raven.move(self.patrol_pos[0]))
                if raven.distance_to(self.patrol_pos[0]) < 1:
                    actions.append(raven.move(self.patrol_pos[1]))
                    self.a=1
            elif self.a==1:
                if raven.distance_to(self.patrol_pos[1]) < 1:
                    actions.append(raven.move(self.patrol_pos[2]))
                    self.a=2
            elif self.a==2:    
                if raven.distance_to(self.patrol_pos[2]) < 1:
                    actions.append(raven.move(self.patrol_pos[3]))
                    self.a=3
            elif self.a==3:
                if raven.distance_to(self.patrol_pos[3]) < 1:
                    actions.append(raven.move(self.patrol_pos[0]))
                    self.a=0

    def reset(self):

        self.front = Point2((self.bot.start_location.x + 10, 30))
        

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ravens = self.bot.units(UnitTypeId.RAVEN)
        
        if ravens.amount == 0 and self.bot.units.tags_in(self.bot.combatArray).amount > 1:
            if self.bot.can_afford(UnitTypeId.RAVEN):
                # 밤까마귀가 하나도 없으면 훈련
                actions.append(cc.train(UnitTypeId.RAVEN))
            if self.bot.die_alert == 2:
                self.bot.die_alert = 1 # 리콘이 죽었다 => 다른곳에서 써먹을 플래그

        elif ravens.amount > 0:
            raven = ravens.first
            # 어디소속이닞 확인
            # 센터 결정해서 patrol_pos 업데이트
            if self.bot.die_alert == 0 or self.bot.die_alert == 1:
                self.bot.die_alert = 2 # 리콘 현재 존재함
            
            for unit in ravens:
                
                if self.bot.units.tags_in(self.bot.combatArray).exists :
                    self.combat_center = self.bot.units.tags_in(self.bot.combatArray).center

                    if self.bot.fighting == 1: #컴뱃유닛들이 싸우고 있으면
                        actions.append(raven.move(self.combat_center)) #컴뱃유닛들의 중앙에서 멈춤

                    else: # 싸우고 있지 않으면 정찰
                        self.patrol(self.combat_center,None)
                        actions.append(raven.move(self.patrol_pos[self.a]))
                
                '''
                # 핵 감지하면 사령부 주위로 돌아가기
                if self.bot.nuke_alert and self.bot.time - self.bot.nuke_time > 14 or self.is_ghost == 2:
                    print("들어오니???????")
                    self.bot.nuke_alert = False

                if self.bot.nuke_alert: 
                    print("들어오니????")
                    self.patrol(self.bot.start_location,self.combat_center.x)
                    actions.append(raven.move(self.patrol_pos[self.a]))
                # 나중에 combat 으로 바꾸기
                else:  '''
                    
                
                self.recon(unit,actions)
                #print(self.bot.is_ghost)

            threaten = self.bot.known_enemy_units.closer_than(10, raven.position) 
            enemy_ghost = threaten(UnitTypeId.GHOST)
            enemy_banshee = threaten(UnitTypeId.BANSHEE)
            
            #assign 관련 플래그 설정
            if raven.distance_to(self.front) > 5: #정면방향이 아님
                if threaten.exists:
                    if threaten.amount > 5:
                        self.bot.have_to_go = 2 #적이 많으면 컴뱃 전체가 움직임
                    elif threaten.amount <= 5:
                        self.bot.have_to_go = 1 #적이 조금이면 컴뱃 일부가 움직임  
                    elif threaten.amount == 1: #적이 하나면 던지고 도망가면되니까 컴뱃x
                        self.bot.have_to_go = 0
                else: self.bot.have_to_go = 0 #적이 없으면 안움직이기(돌아가게)        

            if threaten.exists :
                target = threaten.closest_to(raven.position)
                self.bot.last_pos = target.position
                #print(self.bot.last_pos)
                
                if enemy_ghost.amount > 0 :
                    if self.bot.nuke_alert:
                        self.bot.is_ghost = 2
                    #print(unit.amount)
                    else:
                        self.bot.is_ghost = 1 # 핵 쏘러 옴
                    target = enemy_ghost.first
                    self.bot.last_pos = target.position
                    #print(self.bot.last_pos)
                elif enemy_banshee.amount > 0:
                    target = enemy_banshee.first
                    self.bot.is_ghost = 2

                if raven.distance_to(self.front) > 5 or self.bot.is_ghost > 0 or enemy_banshee.amount > 0: #정면방향이 아니거나, 고스트 혹은 밴시가 있을경우만 공격
                    self.bot.enemy_alert=1 # 에너미 존재
                    self.bot.last_pos = target.position
                    pos = raven.position.towards(target.position, 5)
                    pos = await self.bot.find_placement(UnitTypeId.AUTOTURRET, pos)
                    actions.append(raven(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos))

                    #해병부대 - 열명 넘는지는 어사인에서 관리
                    recon_units = self.bot.units.tags_in(self.bot.reconArray)
                    for unit in recon_units:
                        if unit.type_id == UnitTypeId.MARINE:
                            actions.append(unit.attack(target))

            elif threaten.amount == 0 and self.bot.enemy_alert==1: #평범하게 적들 해치운 경우
                self.bot.enemy_alert=0 # 에너미 해치움
                actions.append(raven.move(self.patrol_pos[self.a]))
                if self.bot.is_ghost > 0:
                    # print("유령해치움")
                    self.bot.is_ghost = 0
                #else :
                    # print("에너미해치움")
                        
            elif self.bot.is_ghost > 0: # 정면방향 고스트였을경우
                if enemy_ghost.amount == 0 and enemy_banshee == 0:
                    self.bot.is_ghost = 0
                    self.bot.enemy_alert=0
                    actions.append(raven.move(self.patrol_pos[self.a]))
            
        return actions


class CombatStrategy(Enum):
    OFFENSE = 0
    DEFENSE = 1


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()  
        self.move_check = 0 #전부 이동완료=1, 아니면 0 
        self.move_tank = 0 #이동 완료한 탱크
        self.combat_pos = 0 #combat_units의 위치
        self.target_pos = self.bot.cc #이동 위치의 기준
        self.position_list = list() #탱크 유닛들의 포지션 리스트

        self.i = 0 #x쪽 가중치 안씀
        self.j = 0 #y쪽 가중치 안씀
        self.n = 0

        if self.bot.start_location.x == 32.5: self.player = 1
        else: self.player = 2


    def distance(self, pos1, pos2):
        """
        두 점 사이의 거리
        """
        result = math.sqrt( math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2))
        return result

        
    def circle(self, target_pos):
        """
        target_pos를 중점으로해서 지그재그 배치
        circle2로 변경해서 안씀
        """
        
        self.position_list.append(Point2((target_pos.x + self.i, target_pos.y + self.j)))
        if(self.n%2 == 1):
            if self.player == 1:
                self.i = self.i + 1 #x좌표는 홀수번째마다 증가
            else:
                self.i = self.i - 1 #x좌표는 홀수번째마다 감소
            self.j = self.j + self.n #y좌표는 홀수에 증가, 짝수에 감소
        else:
            self.j = self.j - self.n


    def circle2(self, target_pos):
        """
        target_pos를 중점으로해서 지그재그 배치
        """
        x=list()
        y=list()
        position=list()
        self.position_list = list()
        r=7

        for theta in range(0, 360):
            if theta % 15 == 0:
                x.append(target_pos.x+r*math.cos(math.radians(theta)))
                y.append(target_pos.y+r*math.sin(math.radians(theta)))

        for i in range(0, len(x)):
            position.append((x[i], y[i])) #x와 y를 짝맞춰 넣어

        if self.player == 1:
            position = sorted(position, key=lambda x: x[0]) #sort by x(position이 (x,y)로 구성됨, 그 중 x로 정렬)
        else:
            position = sorted(position, key=lambda x: x[0], reverse=True) #sort by x reversed

        #print("position: ",position)

        for i in range(0, len(position)):
            self.position_list.append(Point2((position[i][0], position[i][1])))
    
            
    def moving(self, unit, pos, actions):
        """
        전차의 이동 및 변신 담당
        """
        t=0
        self.circle2(pos)
        #if len(self.position_list) == self.n : #리스트 갱신
            #print(len(self.position_list),"@@@@@@@@@",self.n)
            #self.circle2(pos)
        for tank in self.bot.tankArray: #이동 
                #if unit.tag == tank and t<=self.n:
            if unit.tag == tank:
                actions.append(unit.move(self.position_list[t])) 
                break
                    #print("!!!!!!!!!!!!!!!!!")
            t+=1
        if self.distance(unit.position, self.position_list[self.n]) < 1: #지정위치도착
            if unit.type_id == UnitTypeId.SIEGETANK: 
                order = unit(AbilityId.SIEGEMODE_SIEGEMODE) #변신함
                actions.append(order)
        self.n = self.n + 1 
        
    #region Description
    async def step(self):
        actions = list()

        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        mule = self.bot.units(UnitTypeId.MULE)
        mule_pos = self.bot.cc.position.towards(enemy_cc.position, -5)
        #다친 기계 유닛(탱크)
        wounded_units = self.bot.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 0.5
        )  # 체력이 100% 이하인 유닛 검색

        #얼마나 많은 유닛이 공격중인가?
        attacking_units = self.bot.combat_units.filter(
            lambda u: u.is_attacking 
            and u.order_target in self.bot.known_enemy_units.tags
        )  # 유닛을 상대로 공격중인 유닛 검색

        
        ##-----플래그 변경-----
        #3명 이상 공격중이면 fighting인걸로
        if attacking_units.amount > 5:
            self.bot.fighting = 1
        else: self.bot.fighting = 0

        ##-----Mule 기계 유닛 힐-----
        if self.bot.cc.health_percentage < 0.3: #cc피가 우선
            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_pos))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, self.bot.cc))
        elif wounded_units.exists: #아픈 토르가 있는가?
            #공격중인 토르가 있는가?
            wounded_tank = wounded_units.filter(
                lambda u: u.is_attacking 
                and u.order_target in self.bot.known_enemy_units.tags) 

            if wounded_tank.amount > 0:
                wounded_unit =wounded_tank.closest_to(self.bot.cc) #'공격중인' 탱크있으면 걔 우선
            else: wounded_unit =wounded_units.closest_to(self.bot.cc) #없으면 cc에 가까운 탱크 우선

            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, wounded_unit.position))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
        else: #택틱이 뮬인데 치료할거리가 없을경우(치료 끝난 후)
            if mule.amount > 0:
                mule_unit=mule.random
                if self.bot.combat_units.exists:
                    actions.append(mule_unit.move(self.bot.combat_units.center))
                else: actions.append(mule_unit.move(mule_pos))

    #endregion
        
         #방어시 집결 위치
        #defense_position = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos1 = self.bot.start_location + 0.30 * (enemy_cc.position - self.bot.start_location)
        def_pos2 = self.bot.start_location + 0.55 * (enemy_cc.position - self.bot.start_location)
        def_pos3 = self.bot.start_location + 0.75 * (enemy_cc.position - self.bot.start_location)
        def_pos4 = self.bot.start_location + 0.9 * (enemy_cc.position - self.bot.start_location)
        
        _pos1 = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        _pos2 = self.bot.start_location + 0.5 * (enemy_cc.position - self.bot.start_location)
        _pos3 = self.bot.start_location + 0.7 * (enemy_cc.position - self.bot.start_location)
        _pos4 = self.bot.start_location + 0.85 * (enemy_cc.position - self.bot.start_location)

        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        self.n = 0
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            if unit in self.bot.combat_units:
                ##-----타겟 설정-----
                enemy_unit = self.bot.enemy_start_locations[0]
                if self.bot.known_enemy_units.exists:
                    enemy_unit = self.bot.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
                if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                    target = enemy_cc
                else:
                    target = enemy_unit

#region marine
                ##-----전투 유닛 전체-----
                if unit.type_id is UnitTypeId.MARINE:
                    if self.move_check == 0: #초기상태+대기중
                        #print("해병: 초기상태+대기중")
                        self.target_pos = _pos1
                        actions.append(unit.attack(self.target_pos))
                    elif self.move_check == 1: 
                        self.target_pos = _pos2
                        actions.append(unit.attack(self.target_pos))
                    elif self.move_check == 2: 
                        self.target_pos = _pos3
                        actions.append(unit.attack(self.target_pos))
                    elif self.move_check == 3:
                        self.target_pos = _pos4
                        actions.append(unit.attack(self.target_pos))
                    if self.distance(self.bot.combat_units.center, _pos4) < 2: #위치3도착
                        actions.append(unit.attack(target))

                    ##-----스킬-----
                    if self.combat_pos == 1 and unit.distance_to(target) < 15:
                            # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time

                    ##-----타겟-----
                    #타겟 우선순위 설정
                    if self.bot.known_enemy_units.exists:
                        flying_target = self.bot.known_enemy_units.filter(
                            lambda unit:  unit.is_flying
                        )
                        if flying_target.exists:
                            target = flying_target.closest_to(unit)

#endregion
                    ##-----탱크-----
                if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):

                    if self.move_check == 0: #초기상태+대기중
                        self.moving(unit, def_pos1, actions)
                    elif self.move_check == 1: 
                        self.moving(unit, def_pos2, actions)
                    elif self.move_check == 2: 
                        self.moving(unit, def_pos3, actions)
                    elif self.move_check == 3: 
                        self.moving(unit, def_pos4, actions)


        ##-----move_check 변화-----
        if self.bot.combat_units.amount > 40 and self.move_check == 0 and self.bot.tank_units.amount > 3:
            #print("20넘음")
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            #self.i = 0
            #self.j = 0
            self.move_check = 1
            #print("무브체크", self.move_check)
        elif self.bot.combat_units.amount > 80 and self.move_check == 1 and self.bot.tank_units.amount > 7:
            #print("40넘음")
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            #self.i = 0
            #self.j = 0
            self.move_check = 2
            #print("무브체크", self.move_check)
        elif self.bot.combat_units.amount > 100 and self.move_check == 2 and self.bot.tank_units.amount > 10:
            #print("40넘음")
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            #self.i = 0
            #self.j = 0
            self.move_check = 3
            #print("무브체크", self.move_check)
        elif (self.bot.combat_units.amount <= 30 or self.bot.tank_units.amount < 3) and self.move_check >= 1:
            #한타 졌거나 해서 인원 줄었음
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            #self.i = 0
            #self.j = 0
            self.move_check -= 1
            #print("졌음-무브체크", self.move_check)
        elif self.bot.combat_units.amount <= 3 and self.move_check >= 1:
            #한타 졌거나 해서 인원 줄었음
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            #self.i = 0
            #self.j = 0
            self.move_check == 0
            #print("졌음-무브체크", self.move_check)
                          
        #print("move_check: ", self.move_check)

        return actions


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = -1
        # 마지막으로 on_step이 호출된 게임 시간
        self.last_game_time_step_evoked = 0.0 

    def invalid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        elapsed_time = self.bot.time - self.last_game_time_step_evoked
        if elapsed_time < self.seconds_per_step:
            return True
        else:
            # print(C.blue(f'man_step_time: {elapsed_time}'))
            self.step += 1
            self.last_game_time_step_evoked = self.bot.time
            return False

'''
class ProductManager(object):
    """
    어떤 (부대의) 유닛을 생산할 것인지 결정하는 매니저(선택-생산-배치)
    ProductStrategy만들면서 잠깐 안씀!!!
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    async def product(self, next_unit):
        # print("product 불러지면 안되는데 불러짐!!")
        """
        각 부대에서 필요한 유닛을 리스트로 bot에게 줌(비율은 각 매니저에서 계산)
        [combat, recon, nuke] 꼴, 매번 갱신
        이걸 보고 이중에서 어떤걸 생산할지 고르고 생산, 배치를 담당하는 매니저
        어떤걸 우선적으로 고를지는 택틱에 따라 결정
        """
        actions = list()
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        #print(self.bot.ghost_ready)
        #핵 우선생산
        if self.bot.ghost_ready:
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(self.bot.cc(AbilityId.BUILD_NUKE))
                self.bot.ghost_ready = False #고스트는 핵쏘는 중이라 준비x
        #핵 생산할거 없으면 나머지 생산        
        elif self.bot.can_afford(next_unit):
            #print("00000생산 가능")
            if self.bot.time - self.bot.evoked.get((self.bot.cc.tag, 'train'), 0) > 1.0:
                #print("00000마지막 명령을 발행한지 1초 이상 지났음")
                actions.append(self.bot.cc.train(next_unit))
                #print("00000생산명령-----:", next_unit)
                #self.bot.productorder = self.bot.productorder + 1
                #print("생산명령횟수: ", self.bot.productorder)
                self.bot.evoked[(self.bot.cc.tag, 'train')] = self.bot.time
        
        return actions     '''
                     
'''
class RatioManager(object):
    """
    부대에 따라 생성할 유닛 비율 결정하는 매니저
    ProductStrategy만들면서 잠깐 안씀!!!
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai

    def next_unit_select(self, unit_counts, target_unit_counts):
        """
        비율에 따라 next_unit 리턴
        """
        print("ratio-next_unit_select 불러지면 안되는데 불러짐!!")
        target_units = np.array(list(target_unit_counts.values()))
        target_unit_ratio = target_units / (target_units.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
        return next_unit

    def ratio(self):
        """
        부대별 비율에 따라 다음에 생산할 유닛 bot의 next_unit에 저장
        """
        # print("ratio 불러지면 안되는데 불러짐!!")
        #combat_unit_counts = dict()
        recon_unit_counts = dict()
        #nuke_unit_counts = dict()

        #COMBAT
        if self.bot.vespene > 200:
            combat_next_unit = UnitTypeId.SIEGETANK
            #combat_next_unit = UnitTypeId.THOR
        else:
            combat_next_unit = UnitTypeId.MARINE
        
        #RECON
        self.recon_target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 2,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:1,
                UnitTypeId.BATTLECRUISER: 0,
            }
        for unit in self.bot.units:
            if unit.tag in self.bot.reconArray: # 유닛의 태그가 어레이에 포함되어있으면
                recon_unit_counts[unit.type_id] = recon_unit_counts.get(unit.type_id, 0) + 1
        
        recon_next_unit = self.next_unit_select(recon_unit_counts, self.recon_target_unit_counts)

        #self.bot.trainOrder = [combat_next_unit, recon_next_unit, nuke_next_unit]
        #trainOrder = [combat_next_unit, recon_next_unit]
        #유닛 결정
        #if self.bot.product_strategy == ProductStrategy.ATTACK: #combat생산
        #    self.bot.next_unit = trainOrder[0]
        #elif self.bot.product_strategy == ProductStrategy.RECON: #recon생산
        #    self.bot.next_unit = trainOrder[1]  
        #print("11111생산요청: ", self.bot.trainOrder)'''
     

class TrainManager(object):
    """
    CombatManager의 생산 유닛을 결정
    유령과 밤까마귀는 각 매니저에서 생산 결정
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    def next_unit(self):
        if self.bot.vespene > 200 and self.bot.units(UnitTypeId.RAVEN).exists :
            next_unit = UnitTypeId.SIEGETANK
        else:
            next_unit = UnitTypeId.MARINE

        return next_unit


class AssignManager(object): #뜯어고쳐야함
    """
    유닛을 부대에 배치하는 매니저
    *reconArray: recon에 들어가는 해병만 있음
    *nukeArray: 고스트만 있음
    *combatArray: combat의 해병, 탱크, 밤까마귀 있음

    *combat_units: combat의 해병, 탱크만 있음(combatArray보다 작은 개념)
    *nuke_units: 고스트만 있음=nukeArray
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self):
        """
        유닛 타입을 보고 부대별 array에 유닛 배치
        """
        #print("싸우고 있나?: ", self.bot.fighting)
        units_tag = self.bot.units.tags #전체유닛
        
        #and연산으로 살아있는 유닛으로만 구성
        self.bot.combatArray = self.bot.combatArray & units_tag 
        self.bot.reconArray = self.bot.reconArray & units_tag
        self.bot.nukeArray = self.bot.nukeArray & units_tag
        #탱크는 리스트라 다르게
        self.bot.tankArray.clear()
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit.type_id in (UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK):
                self.bot.tankArray.append(unit.tag)


        #이미 할당된 유닛의 태그 빼고
        units_tag = units_tag - self.bot.combatArray - self.bot.reconArray - self.bot.nukeArray


        #------유닛 타입에 따라 array 배정-----
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit.type_id is UnitTypeId.RAVEN: #레이븐은 pass
                self.bot.reconArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.NUKE: #핵은 pass
                pass
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 nuke
                self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.SIEGETANKSIEGED,  UnitTypeId.SIEGETANK): #탱크(변신)는 컴뱃
                self.bot.combatArray.add(unit.tag)
                self.bot.tankArray.append(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)


        ##------플래그에 따라 array 배정------
        #print("fighting: ", self.bot.fighting, "/ have_togo: ", self.bot.have_to_go)
        #print("fighting이 0이고 have_togo가 1,2면 이동, fighting이 1이면 돌아옴")
        #print("nuke:", len(self.bot.nukeArray))
        #print("recon:", len(self.bot.reconArray))
        #print("combat:", len(self.bot.combatArray))
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        #nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
        #combat_units = self.bot.units.tags_in(self.bot.combatArray)
        #if len(self.bot.combatArray) >= 10:
            #rint("에러확인1:",combat_units(UnitTypeId.MARINE).closest_to(cc).tag)
        #if len(self.bot.nukeArray) >= 1:
            #print("에러확인2:",nuke_units(UnitTypeId.MARINE).amount)

        '''
        if self.bot.fighting == 0: #컴뱃이 싸우고 있지 않음
            #------recon-----
            if self.bot.have_to_go == 1: #일부만 와라
                if len(self.bot.combatArray) >= 10: #그때 컴뱃에 10명은 넘어야(중간에 지켜야해서)
                    while len(self.bot.reconArray) == 5: #recon이 5명 될때까지
                        tag = self.bot.combat_units(UnitTypeId.MARINE).closest_to(cc).tag #해병중 cc에 가까운애 골라내기
                        self.bot.reconArray.add(tag) #recon으로 이동
                        self.bot.combatArray.remove(tag)
                        #print("이동완료1", len(self.bot.reconArray), "/", len(self.bot.combatArray))
                """
                #해병 5명 뽑기
                for tag in self.bot.combatArray:
                    unit = self.bot.units.find_by_tag(tag)
                    if unit.type_id is UnitTypeId.MARINE: #해병인가?
                        if len(self.bot.reconArray) <=5: #5명 이내인가?
                            self.bot.reconArray.add(unit.tag) #그럼 레콘으로 이동
                            break"""

            elif self.bot.have_to_go == 2: #전체가 와라
                for tag in self.bot.combatArray:
                    self.bot.reconArray.add(tag) #combat의 전부를 recon으로
                    #print("이동완료2", len(self.bot.reconArray), "/", len(self.bot.combatArray)) 
                self.bot.combatArray.clear() #그리고 combat 전체삭제

            #-----nuke-----
            #고스트가 새로 생성되어야하고 + 일단 윗길 아랫길만 해둠
            if self.bot.units(UnitTypeId.GHOST).amount == 0 and self.bot.nuke_strategy<=1:
                if self.bot.die_count <= 3: #combat이 defense인 상태
                    if len(self.bot.combatArray) >= 10: #그때 컴뱃에 10명은 넘어야(중간에 지켜야해서)
                        while self.bot.nuke_units(UnitTypeId.MARINE).amount == 5:
                            tag = self.bot.combatArray.pop() #랜덤하게 뽑음(제거)
                            self.bot.nukeArray.add(tag) #nuke로 이동
            
    
        elif self.bot.fighting == 1: #컴뱃이 싸우고 있음
            if len(self.bot.reconArray) >= 5: #몇명이 빠져있으면 돌아와라
                for tag in self.bot.reconArray:
                    unit = self.bot.units.find_by_tag(tag)
                    if unit.type_id is UnitTypeId.MARINE:
                        tag = self.bot.reconArray.pop() #뽑음(제거)
                        self.bot.combatArray.add(tag) #combat으로
                #print("완전돌아옴", len(self.bot.reconArray),"/", len(self.bot.combatArray))
            if self.bot.nuke_units(UnitTypeId.MARINE).amount >= 5: #몇명이 빠져있으면 돌아와라
                for tag in self.bot.nukeArray:
                    unit = self.bot.units.find_by_tag(tag)
                    if unit.type_id in (UnitTypeId.MARINE):
                        tag = self.bot.nukeArray.pop() #뽑음(제거)
                        self.bot.combatArray.add(tag)'''
                
        


        """
        #유닛 타입에 따라 array 배정
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit is None:
                pass
            #elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 레콘으로
                #self.bot.reconArray.add(unit.tag)
            #elif unit.type_id is UnitTypeId.GHOST: #고스트는 누크로
                #self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP): #토르는 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            else: #아무것도 아니여도 일단 컴뱃으로
                self.bot.combatArray.add(unit.tag)"""

        """
        if self.bot.product_strategy == ProductStrategy.ATTACK :
            self.bot.combatArray = self.bot.combatArray | units_tag
        elif self.bot.product_strategy == ProductStrategy.RECON:
            self.bot.reconArray = self.bot.reconArray | units_tag"""
        #elif self.bot.product_strategy == ProductStrategy.NUKE:
            #self.bot.nukeArray = self.bot.nukeArray | units_tag

        ## print("assign됨--------")

    def reassign(self):
        """
        이미 배정된 유닛을 다시 배정
        있을리 없는 유닛타입을 옮겨줌
        """
        for tag in self.bot.combatArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 컴뱃태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 삭제
                self.bot.combatArray.remove(unit.tag)
                self.bot.reconArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 옮기기
                self.bot.combatArray.remove(unit.tag)
                self.bot.nukeArray.add(unit.tag)

        for tag in self.bot.nukeArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 누크태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 삭제
                self.bot.nukeArray.remove(unit.tag)
                self.bot.reconArray.add(unit.tag)
            elif unit.type_id in [UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK]: #탱크는 옮기기
                self.bot.nukeArray.remove(unit.tag)
                self.bot.combatArray.add(unit.tag)

        for tag in self.bot.reconArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 레콘태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 옮기기
                self.bot.reconArray.remove(unit.tag)
                self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in [UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK]: #탱크는 옮기기
                self.bot.reconArray.remove(tag)
                self.bot.combatArray.add(unit.tag)

             

        ## print("reassign됨--------")


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    공격 타이밍은 강화학습을 통해 학습
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc() 

        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.assign_manager = AssignManager(self)
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        #self.ratio_manager = RatioManager(self)
        self.train_manager = TrainManager(self)
        #self.product_manager = ProductManager(self)
        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()
        self.tankArray = list()

        self.nuke_reward = 0 
        self.nuke_strategy= 0
        self.combat_strategy = 0
        #self.trainOrder=list()
        #self.next_unit = UnitTypeId.MARINE
        

        #self.nukeGo = 0 #핵 쏜 횟수-bigDamage있으면 일단 필요없음 그냥 확인용으로 남겨둠
        #self.previous_Damage = 0 #이전 틱의 토탈대미지(아래를 계산하기 위한 기록용, 매 틱 갱신)
        #self.bigDamage = 0 #한번에 500이상의 큰 대미지가 들어간 횟수(핵이 500이상)

        # 정찰부대에서 사용하는 플래그
        self.threaten=list()
        self.enemy_alert=0
        self.die_alert=0 # 레이븐 죽었을때
        self.is_ghost=0 # 적이 유령인지
        self.last_pos=(0,0) # 레이븐 실시간 위치

        # 핵 부대에서 사용하는 플래그
        self.die_count=0
        self.ghost_ready = False
        
        #self.productorder = 0 #생산명령 들어간 횟수
        #self.productdone = 0 #생산명령 수행 횟수
        self.productIng = 0 #생산명령들어가면 1, 처리되면 0

        # assign에서 사용하는 플래그(갱신은 컴뱃, 레콘에서 함)
        self.fighting = 0 #현재 컴뱃에서 싸우고 있는가? 0:No, 1:Yes
        self.have_to_go = 0 #레콘으로 옮겨야하는가? 0:No, 1:일부만, 2:전체

        self.nuke_alert = False
        self.nuke_time = 0
        
    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.combat_strategy = CombatStrategy.DEFENSE
        self.nuke_strategy = 2 #0,1,2,3
        #self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE, None] #Combat,Recon,Nuke
        #self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE] #Combat,Recon,Nuke
        self.evoked = dict() #생산명령 시간 체크

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()
        self.nuke_manager.reset()
        self.recon_manager.reset()
        
        self.assign_manager.assign()


        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌
        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        '''
        if self.alert(Alert.NuclearLaunchDetected):
            print("핵이 준비되었습니다@@@@@@@@@")
            self.nuke_alert = True
            self.nuke_time = self.time
        '''

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록
        #부대별 units타입
        
        #self.combat_units = self.units.tags_in(self.bot.combatArray)
        '''
        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경
            before = self.nuke_strategy
            if self.combat_units.amount > 20:
                if self.tank_units.amount >= 3:
                    self.combat_strategy = CombatStrategy(0) #OFFENSE
                    self.combat_manager.move_check = 0
            self.nuke_strategy = self.set_strategy()
            if self.units(UnitTypeId.GHOST).amount > 0:
                if before != self.nuke_strategy:
                    self.nuke_reward -= 0.001
                else :
                    self.nuke_reward += 0.001
            ## print("-------생산택틱: ", self.product_strategy)
            #nuke reward 초기화
            self.nuke_reward = 0
        ''' # 고스트만 강화학습하는 가정 하에 해봄

        '''
        if self.time - self.last_step_time >= self.step_interval and self.units(UnitTypeId.GHOST).amount == 0:
            self.nuke_strategy = self.set_strategy()
        '''
        #self.assign_manager.reassign() #이상하게 배치된 경우 있으면 제배치

        """
        if self.state.score.total_damage_dealt_life - self.previous_Damage > 500: #한번에 500이상의 대미지를 주었다면
            self.bigDamage += 1
            # print("한방딜 ㄱ: ", self.bigDamage, "딜량: ", self.state.score.total_damage_dealt_life - self.previous_Damage)
        self.previous_Damage = self.state.score.total_damage_dealt_life #갱신 
        """
        #self.last_step_time = self.time
        
            

        #생산 명령이 처리되었다면
        if self.productIng == 0: 
            #self.ratio_manager.ratio() #next_unit 바꿔주고

            #생산
            #actions += await self.product_manager.product(self.next_unit) #생산
            actions += await self.train_action() #생산
            self.productIng = 1 #생산명령 들어갔다고 바꿔줌

        #생산 명령이 들어갔다면
        elif self.productIng == 1:
            #대기
            #await self.on_unit_created(self.next_unit) #명령넣은게 생산될때까지 기다림
            #self.productdone = self.productdone + 1
            #print("생산명령 수행횟수: ", self.productdone) #엄청큼 문제있음
            ## print(self.next_unit)

            #배치
            self.assign_manager.assign()
            self.productIng = 0 #생산명령 수행했다고 바꿔줌 

        #밤까마귀는 train action에서 담당하지 않아서 밤까마귀가 없으면 배치가 안됨
        #따라서 그냥 한번 더 해줌... 수정 필요
        self.assign_manager.assign()

        self.combat_units = self.units.filter(
            lambda unit: unit.tag in self.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )

        self.nuke_units = self.units.tags_in(self.nukeArray)
        
        self.tank_units = self.combat_units.filter(
            lambda unit:  unit.type_id in [UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        
        #print("컴뱃어레이: ", len(self.combatArray), "컴뱃유닛츠: ", self.combat_units.amount)

        #print(self.nuke_strategy)
        #actions += await self.attack_team_manager.step()
        
        #actions += await self.product_manager.product(self.next_unit) #생산
        actions += await self.recon_manager.step() 
        actions += await self.combat_manager.step()   
        
        actions += await self.nuke_manager.step()
        
        ## -----명령 실행-----
        await self.do_actions(actions)
        ## print(self.on_unit_created)
        
        ## print("한거: ",actions)


    async def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()
        next_unit = self.train_manager.next_unit()

        cc_abilities = await self.get_available_abilities(self.cc)
        #핵 우선생산
        
        # 이미 핵이 있으면 생산X
        if self.units(UnitTypeId.NUKE).amount > 0 and self.ghost_ready:
            self.ghost_ready = False
        if self.ghost_ready:
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(self.cc(AbilityId.BUILD_NUKE))
                self.ghost_ready = False #고스트는 핵쏘는 중이라 준비x
        elif self.can_afford(next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                if self.units.tags_in(self.combatArray).amount > 15 and self.units(UnitTypeId.RAVEN).amount == 0 :
                    pass
                else: 
                    actions.append(self.cc.train(next_unit))
                    self.evoked[(self.cc.tag, 'train')] = self.time
        return actions


    def set_strategy(self):
        #
        # 특징 추출
        #
        state = np.zeros(5, dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        #state[5] = self.nuke_reward
        '''
        state[6] = len(self.combatArray) #combat 부대의 유닛 수 - combat결정용 state
        for unit in self.units.not_structure:
            if unit in self.units.tags_in(self.combatArray):
                if unit.type_id is not UnitTypeId.MULE:
                    #토르 변신중이여도 토르로 취급
                    if unit.type_id is UnitTypeId.THORAP:
                        state[6 + ProductStrategy.to_index[UnitTypeId.THOR]] += 1
                    else:
                        state[6 + ProductStrategy.to_index[unit.type_id]] += 1'''
        #state[6] = self.trainOrder #각 부대별 생산요청 유닛 - product 결정용 - 리스트라 안됨
        state = state.reshape(1, -1)

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        if self.sock is not None:
            self.sock.send_multipart(data)
            data = self.sock.recv_multipart()
            value = pickle.loads(data[0])
            action = pickle.loads(data[1])
        else:
            with torch.no_grad():
                value, logp = self.model(torch.FloatTensor(state))
                value = value.item()
                action = logp.exp().multinomial(num_samples=1).item()
        #combat_strategy = CombatStrategy.to_type_id[action // len(NukeStrategy)]
        nuke_strategy = NukeStrategy(action)
        #nuke_strategy = action % len(NukeStrategy)
        # product_strategy = ProductStrategy.to_type_id(action // (len(NukeStrategy)*len(mule_strategy)))
        # nuke_strategy = NukeStrategy(action // len(NukeStrategy))
        # mule_strategy = MuleStrategy(action % len(MuleStrategy))
        return nuke_strategy.value


    def on_end(self, game_result):
        if self.sock is not None:
            #score = 1. if game_result is Result.Victory else -1.
            
            if game_result is Result.Victory:
                score = 0.5
            else: score = -0.5
            if self.nuke_reward > 0:  #한번에 큰 대미지 넣은 횟수만큼
                score += self.nuke_reward
            # print("리워드: ", score)
            self.sock.send_multipart((
                CommandType.SCORE, 
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()



    '''def on_end(self, game_result):
        for tag in self.combatArray:
            # print("111전투 : ", tag)
        # print("-----")
        for tag in self.reconArray:
            # print("2222222정찰 : ", tag)
        # print("-----")
        for tag in self.nukeArray:
            # print("핵 : ", tag)
        # print("-----")'''
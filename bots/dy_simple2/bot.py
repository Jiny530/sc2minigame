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
from sc2.ids.effect_id import EffectId
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

#from .consts import CommandType, NukeStrategy


"""
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
        return value, logp"""

    
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
        self.is_nuke=0
        self.nuke_target = None
        self.run = 0
        self.is_raven = 0

    def reset(self):
        self.ghost_pos = self.bot.start_location.x
        self.enemy_pos= self.bot.enemy_cc.x
        self.course = [Point2((self.ghost_pos,55)),Point2((self.enemy_pos,55)),Point2((self.ghost_pos,10)),Point2((self.enemy_pos,10))]
        if self.ghost_pos == 32.5:
            self.middle = 72.5
        else:
            self.middle = 52.5

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        # 고스트 생산명령 내릴준비
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
        
        # 2분뒤에 생성

        #if ghosts.amount == 0 and self.bot.die_count < 2 :
            #if self.bot.time >= 200:
                #actions.append(cc.train(UnitTypeId.GHOST))
                #self.bot.ghost_ready = True # 핵 항상 생산

        if ghosts.amount > 0:

            if self.dead == 0 and self.bot.nuke_strategy == 0: # 위에서 죽었는데 또 위로가면 -> 학습 안할거면 수정
                self.bot.nuke_reward -= 0.01 #마이너스
            elif self.dead == 1 and self.bot.nuke_strategy == 1:
                self.bot.nuke_reward -= 0.01 #마이너스
            elif self.dead == 2 and self.bot.nuke_strategy == 2:
                self.bot.nuke_reward -= 0.01 #마이너스

            self.dead = 3 

            ghost = ghosts.first #고스트는 딱 한 개체만
            ghost_abilities = await self.bot.get_available_abilities(ghost)
            threaten = self.bot.known_enemy_units.closer_than(15, ghost.position)
            nuke_units = self.bot.units.tags_in(self.bot.nukeArray)
            ravens = threaten(UnitTypeId.RAVEN)
            
            

            # 위로
            if self.bot.nuke_strategy == 0 and self.pos != 3: 
                for unit in nuke_units:
                # 위치 이동
                    if unit.distance_to(self.course[0]) > 3 and self.pos == 0:
                        actions.append(unit.move(self.course[0]))
                        self.pos=1 # 올라가기

                    if self.run or (unit.distance_to(self.course[0]) == 0 and self.pos == 1):
                        self.run = 0
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
                        self.pos=1 # 내려가기

                    if self.run or (unit.distance_to(self.course[2]) == 0 and self.pos == 1):
                        self.run = 0
                        actions.append(unit.move(self.course[3]))
                        self.pos=2 # 옆으로 가기

                    if unit.distance_to(self.course[3]) < 3:
                        self.pos=3 # 대기장소 도착

            # 가운데로 가라 - 유령 혼자만 갈 거임
            elif self.bot.nuke_strategy == 2 : 
                
                
                # 에너지가 50 이상일때, 가운데로 출발
                if self.bot.combat_units.exists and ghost.distance_to(self.bot.combat_units.center) < 3:
                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKOFF_GHOST))

                if not self.bot.ghost_ready and AbilityId.BUILD_NUKE not in cc_abilities and ghost.energy > 50 and ghost.distance_to(Point2((self.middle,31.5))) > 3 and self.pos == 0 :
                    actions.append(ghost.move(Point2((self.middle,self.bot.enemy_cc.y)))) #중간지점으로 가기
                    self.pos=1 

                if self.pos == 1 and ghost.distance_to(Point2((self.middle,31.5))) < 3 and ghost.energy < 15:
                    self.pos = 0
                    #self.is_nuke = 0
                    self.bot.ghost_ready = False
                    if self.bot.combat_units.exists:
                        actions.append(ghost.move(self.bot.combat_units.center))
                    else:
                        actions.append(ghost.move(self.bot.start_location))

                if self.pos == 0 and ghost.energy < 25 and self.bot.combat_units.exists:
                    if self.bot.start_location.x < 40:
                        actions.append(ghost.move(Point2((self.bot.combat_units.center.x - 2,31.5))))
                    else:
                        actions.append(ghost.move(Point2((self.bot.combat_units.center.x + 2,31.5))))
                elif self.pos == 0 and ghost.energy < 25:
                    actions.append(ghost.move(self.bot.start_location))

                if threaten.amount > 0 :
                    if ravens.exists:
                        # 레이븐 만나면 일단 다 포기하고 첫 위치로..?
                        self.is_raven = 2
                        self.pos = 0
                        #self.is_nuke = 0
                        self.bot.ghost_ready = False
                        if self.bot.combat_units.exists:
                            actions.append(ghost.move(self.bot.combat_units.center))
                        else:
                            actions.append(ghost.move(self.bot.start_location))
                    
                    # 컴뱃부대 존재하고 가까이에 있으면 핵X
                    if self.bot.combat_units.exists and ghost.distance_to(self.bot.combat_units.center) < 5:
                        if self.pos == 0:
                            if self.bot.start_location.x < 40:
                                actions.append(ghost.move(Point2((self.bot.combat_units.center.x - 2,31.5))))
                            else:
                                actions.append(ghost.move(Point2((self.bot.combat_units.center.x + 2,31.5))))
                        # 인원수 적으면 은신
                        if self.bot.combat_units.amount < 15:
                            actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                        if self.bot.combat_units(UnitTypeId.RAVEN).exists:
                            pass # 밤까마귀 없으면 있을때 인식했던 밴시, 고스트에 EMP탄환 쏘기
                        elif threaten(UnitTypeId.BANSHEE).exists:
                            actions.append(ghost(AbilityId.EMP_EMP, target=threaten(UnitTypeId.BANSHEE).closest_to(ghost).position))
                        elif threaten(UnitTypeId.GHOST).exists:
                            actions.append(ghost(AbilityId.EMP_EMP, target=threaten(UnitTypeId.GHOST).closest_to(ghost).position))

                        if self.is_raven == 0 and ghost.energy > 50:
                            self.bot.ghost_ready = True
                            if self.bot.time - self.nuke_time <= 2 and (ghost.position.x - self.nuke_target.x)*(ghost.position.x - self.bot.enemy_cc.x) > 0:
                                pass # 금방 핵 쐈던 방향으로 걸어가지마 
                            elif AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities:
                                self.bot.ghost_ready = False
                                actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_cc))

                        

                    # 컴뱃부대랑 멀리있으면 적 발견하자마자 은신, 핵 O
                    else: 
                        actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                        
                        # 적이 15명 이상이고 나랑 더 가깝다면 핵 쏘기-> 나중에 확인
                        if threaten.amount > 15: # and ghost.distance_to(threaten.center) < self.bot.combat_units.distance_to(threaten.center):
                            self.bot.ghost_ready = True # 핵 생산해
                            
                            if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghost.energy > 14: 
                                self.bot.ghost_ready = False
                                # 적이랑 가까우면 멀리 떨어져서 쏘기
                                if ghost.distance_to(threaten.center) < 10:
                                    distance = ghost.position.x - threaten.center.x
                                    if distance > 0:
                                        distance = 12 - distance
                                    else:
                                        distance = -12 - distance 
                                    actions.append(ghost.move(Point2((ghost.position.x + distance, ghost.position.y))))
                                else:
                                    if threaten.amount > 15:
                                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=threaten.closest_to(ghost).position))
                                        self.nuke_time = self.bot.time
                                        self.nuke_target = threaten.center
                            # 핵 쏴야하는데 은신에너지 부족하면 후퇴        
                            elif ghost.energy < 14 : 
                                self.pos = 0
                                #self.is_nuke = 0
                                self.bot.ghost_ready = False
                                if self.bot.combat_units.exists:
                                    actions.append(ghost.move(self.bot.combat_units.center))
                                else:
                                    actions.append(ghost.move(self.bot.start_location))
                        
                        # 적이 15명 이상이 아니라면 핵 쏘러가기
                        elif self.pos == 1:
                            if ghost.energy >= 14 and ghost.is_cloaked:
                                self.bot.ghost_ready = True
                                if self.bot.time - self.nuke_time <= 2 and self.nuke_target != None and (ghost.position.x - self.nuke_target.x)*(ghost.position.x - self.bot.enemy_cc.x) > 0:
                                    pass # 금방 핵 쐈던 방향으로 걸어가지마 
                                elif AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities:
                                    self.bot.ghost_ready = False
                                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_cc))
                            elif ghost.energy < 14: #
                                if self.bot.combat_units.exists:
                                    actions.append(ghost.move(self.bot.combat_units.center))
                                else:
                                    actions.append(ghost.move(self.bot.start_location))
                                self.pos = 0
                                self.bot.ghost_ready = False
                            elif ghost.energy >= 39 :
                                self.bot.ghost_ready = True
                                if self.bot.time - self.nuke_time < 2 and self.nuke_target != None and (ghost.position.x - self.nuke_target.x)*(ghost.position.x - self.bot.enemy_cc.x) > 0:
                                    pass
                                elif AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities:
                                    self.bot.ghost_ready = False
                                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_cc))

                self.bot.ghost_ready = True # 핵 항상 생산
                        
                    
              
            # 위, 아래로 갈때 적 발견 시 행동
        
            if self.bot.nuke_strategy < 2 and threaten.amount > 0 :
                if ravens.amount > 0: # 밤까마귀이면 도망가는 코드 추가
                    self.is_raven = 1
                    '''
                        for unit in nuke_units:
                            if unit.type_id == UnitTypeId.GHOST:
                                pass
                            else:
                                actions.append(unit.attack(ravens.closest_to(unit)))
                    '''
                else :
                    # @@ 수정필요 상대팀이 우리보다 많으면 고스트는 은폐, 핵공격
                    if threaten.amount > 15:
                        actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=threaten.closest_to(ghost).position))
                

            if self.pos==3 or self.stop and ghost.is_idle:
                self.bot.ghost_ready = True # @@고스트 준비됐음 플레그, 핵 우선으로 생산할까??
                ghost_abilities = await self.bot.get_available_abilities(ghost)
                
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghost.is_idle:
                    # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_cc))
                    self.nuke_time = self.bot.time
                    #self.stop = True
                    self.bot.nuke_reward += 0.1

            #은폐중, 에너지가 16보다 적고, 핵이 준비되어있지 않으면 도망치기
                if ghost.is_cloaked and ghost.energy < 14 :
                    self.run = 1
                    self.pos = 1
                    self.is_nuke = 0
                    #self.bot.ghost_ready = False

            if self.bot.combat_units.exists and ghost.distance_to(self.bot.start_location) < self.bot.combat_units.center.distance_to(self.bot.start_location):
                actions.append(ghost(AbilityId.BEHAVIOR_CLOAKOFF_GHOST))
            
        elif ghosts.amount==0 and self.dead==3 : #고스트 죽음 (amount==0)
            self.pos=0
            self.bot.die_count += 1
            self.stop = False
            self.bot.ghost_ready = False
            #self.bot.nuke_reward -= 0.1

            if self.is_raven == 1:
                self.bot.nuke_strategy = 2
            elif self.is_raven == 2:
                self.bot.nuke_strategy = randint(0,1)
            

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
                          
    def patrol(self, center, combat_center):
        # 사령부만 정찰 혹은 combat 만을 정찰
        if combat_center == None:
            if center.x != self.bot.start_location.x : # 컴뱃정찰이면 좁게 정찰
                self.patrol_pos = [Point2((center.x, center.y+7)), Point2((center.x+7, center.y)), Point2((center.x, center.y-7)), Point2((center.x-7, center.y))]
            else:
                self.patrol_pos = [Point2((center.x, center.y+10)), Point2((center.x+10, center.y)), Point2((center.x, center.y-10)), Point2((center.x-10, center.y))]
        
        
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
                    

    async def setAutoturret(self,unit,target,actions,ab):
        if unit.distance_to(target.position) > 5:
            dis = unit.distance_to(target.position) - 5
        else:
            dis = 5
        
        if AbilityId.BUILDAUTOTURRET_AUTOTURRET in ab:
            pos = unit.position.towards(target.position, dis)
            pos = await self.bot.find_placement(UnitTypeId.AUTOTURRET, pos)
            actions.append(unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos))

    def position_set(self,unit,target,offset,actions):
        if self.bot.start_location.x < 40:
            actions.append(unit.move(Point2((target.x - offset, target.y))))
        else:
            actions.append(unit.move(Point2((target.x + offset, target.y))))

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ravens = self.bot.units(UnitTypeId.RAVEN)

        #if not ravens.exists :
            #if (self.bot.nuke_alert and self.bot.command_nuke and self.bot.time - self.bot.nuke_time < 10) or (self.bot.cloak_units.amount > 0):
            # 시간넉넉하면 밤까마귀 생성해서 막기
            # train_action에서 플래그 보고 자원 아껴야함
                #actions.append(cc.train(UnitTypeId.RAVEN))
        '''
        if ravens.amount == 0 and self.bot.combat_units.amount > 10:
            if self.bot.can_afford(UnitTypeId.RAVEN):
                # 밤까마귀가 하나도 없으면 훈련
                actions.append(cc.train(UnitTypeId.RAVEN))
        '''
        if ravens.amount > 0:
            raven = ravens.first            
            
            threaten = self.bot.known_enemy_units.closer_than(15, raven.position)
            cloak_units = threaten.filter(
                lambda unit: unit.type_id is UnitTypeId.GHOST or unit.type_id is UnitTypeId.BANSHEE
            )
            enemy_ghost = threaten(UnitTypeId.GHOST)
            enemy_banshee = threaten(UnitTypeId.BANSHEE)

            mechanic = self.bot.known_enemy_units.filter(
                lambda unit:  unit.is_mechanical
            )

            for unit in ravens:
                
                unit_abilities = await self.bot.get_available_abilities(unit)
                # 핵 감지
                if self.bot.nuke_alert :
                    if self.bot.command_nuke:
                    # 사령부 주위 핵이면

                        if unit.energy > 45:
                            target = self.bot.known_enemy_units.filter(
                                lambda u: u.type_id is UnitTypeId.GHOST and u.distance_to(self.bot.nuke_pos) < 12
                            )
                            
                            if target.exists: # 타겟이 범위안에 있으면 오토터렛 범위에 따라 던지기
                                await self.setAutoturret(unit,target.closest_to(unit),actions,unit_abilities)
                            else: #타겟이 범위 안에 없으면 찾기
                                self.patrol(self.bot.nuke_pos,None)
                                if enemy_ghost.exists:
                                    target = enemy_ghost.filter(
                                        lambda u: u.distance_to(self.bot.nuke_pos) < 12
                                    )
                                    if target.exists:
                                        await self.setAutoturret(unit,target.closest_to(unit),actions,unit_abilities)
                                    else: # 유령은 있는데 핵은 아직 안쐈다?
                                        pass
                                else:        
                                    self.recon(unit,actions)
                    else:
                        run_alert = 0
                        # 사령부 주위 핵이 아니면
                        for u in self.bot.combat_units:
                            if u.distance_to(self.bot.nuke_pos) < 10:
                                run_alert = 1
                        if self.bot.combat_units.exists and run_alert:
                            target = self.bot.known_enemy_units.filter(
                                lambda u: u.type_id is UnitTypeId.GHOST and u.distance_to(self.bot.nuke_pos) < 12
                            )
                            if target.exists:
                                if unit.energy + (11 - self.bot.time + self.bot.nuke_time) >= 50 and unit.distance_to(target) - 4*(11 - self.bot.time + self.bot.nuke_time) < 7 :
                                    await self.setAutoturret(unit,target.closest_to(unit),actions,unit_abilities)
                                else:
                                    self.bot.run_alert = 1
                            else :
                                if self.bot.time - self.bot.nuke_time < 10:
                                    self.patrol(self.bot.nuke_pos,None)
                                    self.recon(unit,actions)
                                else :
                                    self.bot.run_alert = 1
                                    actions.append(unit.move(self.bot.runaway(unit.position,self.nuke_pos,13)))

                elif self.bot.combat_units.exists :
                    combat_center = self.bot.units.tags_in(self.bot.combatArray).center

                    # 밤까마귀를 공격할 수 있는 유닛
                    flying_threaten = threaten.filter(
                        lambda u: u.type_id is UnitTypeId.GHOST or u.type_id is UnitTypeId.VIKINGFIGHTER or u.type_id is UnitTypeId.BATTLECRUISER
                    )

                    # 토르 사거리가 센터에 닿으면
                    if threaten(UnitTypeId.THOR).exists and threaten(UnitTypeId.THOR).closest_to(unit).distance_to(combat_center) < 11:
                        thor = threaten(UnitTypeId.THOR).closest_to(unit)
                        self.position_set(unit,thor.position,11,actions)
                    # 기타 사거리 6인 유닛이 센터에 닿으면
                    elif flying_threaten.exists and flying_threaten.closest_to(unit).distance_to(combat_center) < 7:
                        flying = flying_threaten.closest_to(unit)
                        self.position_set(unit,flying.position,7,actions)
                    else: # 안 닿으면 센터에 있기
                        actions.append(unit.move(combat_center)) 
                    
                    # 가장 가까운 장갑 유닛
                    if mechanic.exists and AbilityId.EFFECT_ANTIARMORMISSILE in unit_abilities:
                        if mechanic(UnitTypeId.BATTLECRUISER).exists:
                            actions.append(unit(AbilityId.EFFECT_ANTIARMORMISSILE, mechanic(UnitTypeId.BATTLECRUISER).closest_to(unit)))
                        else:
                            actions.append(unit(AbilityId.EFFECT_ANTIARMORMISSILE, mechanic.closest_to(unit)))
                    
                    if self.bot.cloak_units.exists:
                        cloak = self.bot.cloak_units.closer_than(15,unit)
                        if cloak.exists:
                            cloak = cloak.closest_to(unit)
                            if unit.distance_to(cloak) > 11:
                                distance = unit.position.x - cloak.position.x
                                if distance > 0 :
                                    distance = distance - 11
                                else:
                                    distance = distance + 11
                                actions.append(unit.move(Point2((unit.position.x - distance,unit.position.y))))

                elif self.bot.cloak_units.exists:
                        cloak = self.bot.cloak_units.closer_than(15,unit)
                        if cloak.exists:
                            cloak = cloak.closest_to(unit)
                            if unit.energy > 45:
                                await self.setAutoturret(unit,cloak,actions,unit_abilities)
                else:
                    if self.bot.start_location.x < 40:
                        actions.append(unit.move(Point2((29, 30))))
                    else:
                        actions.append(unit.move(Point2((98, 30))))
            
        return actions


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()  
        self.move_check = 0 #이동 체크
        self.target_pos = self.bot.start_location #해병 이동 위치의 기준
        self.position_list = list() #탱크 유닛들의 목표 포지션 리스트
        self.marine_call = 1 #탱크가 출발 후 제자리에 도착하면 해병을 부름 0이면 부르기
        self.now_marine = 0 #combat중 해병 수 기록(현재)
        self.before_marine = 0 #combat중 해병 수 기록(과거)
        self.move_time = self.bot.time #무브체크가 바뀌는 타이밍 기록
        self.less_marine = self.bot.time #해병이 15 이하일때를 기록(시간), 25초이상 지속되면 무브체크 감소

        # 파란색 기지일때
        if self.bot.start_location.x > 40:
            self.tank_center = [Point2((80,31.5)),Point2((71,31.5)),Point2((62,31.5)),Point2((53,31.5)),Point2((44,31.5)), Point2((35,31.5))]
            self.marine_center = [Point2((85,31.5)),Point2((76,31.5)),Point2((67,31.5)),Point2((58,31.5)),Point2((49,31.5)), Point2((40,31.5))]
            self.defense_center = Point2((65, 31.5))
        # 빨간색 기지일때
        else:
            self.tank_center = [Point2((46,31.5)),Point2((55,31.5)),Point2((64,31.5)),Point2((73,31.5)),Point2((82,31.5)), Point2((91,31.5))]
            self.marine_center = [Point2((41,31.5)),Point2((50,31.5)),Point2((59,31.5)),Point2((68,31.5)),Point2((77,31.5)), Point2((86,31.5))]
            self.defense_center = Point2((65, 31.5))

    def distance(self, pos1, pos2):
        """
        두 점 사이의 거리
        """
        result = math.sqrt(math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2))
        return result


    def circle2(self, target_pos):
        """
        target_pos를 중점으로해서 원모양으로 배치
        x, y좌표들을 구해서 튜플(x,y)로 변환해 position에 넣음, 그리고 그걸 Point2로 바꿔서 position_list에.
        position_list: [(x,y), (x,y), (x,y)....]
        """
        x=list()
        y=list()
        position=list()
        self.position_list = list() #초기화
        r=7

        for theta in range(0, 360):
            if theta % 17 == 0:
                x.append(target_pos.x+r*math.cos(math.radians(theta)))
                y.append(target_pos.y+r*math.sin(math.radians(theta)))

        for i in range(0, len(x)):
            position.append((x[i], y[i])) #x와 y를 짝맞춰 넣어

        if self.bot.start_location.x < 40:
            position = sorted(position, key=lambda x: x[0]) #sort by x(position이 (x,y)로 구성됨, 그 중 x로 정렬)
        else:
            position = sorted(position, key=lambda x: x[0], reverse=True) #sort by x reversed

        for i in range(0, len(position)):
            self.position_list.append(Point2((position[i][0], position[i][1])))


    def defense_circle(self, unit, center):
        """
        해병의 대기 위치를 계산
        """
        t = 0
        r = 0
        x = 0
        y = 0

        if self.bot.start_location.x < 40: #빨강팀
            theta = [180, 175, 185, 170, 190, 165, 195, 160, 200, 155, 205, 150, 210]
        else: #파랑팀
            theta = [0, 5, 355, 10, 350, 15, 345, 20, 340, 25, 335, 30, 330]

        #마린
        if unit.type_id is UnitTypeId.MARINE:
            for marine in self.bot.marineArray:
                if unit.tag == marine: 
                    if t == 0: 
                        r = 20
                        x = center.x + r*math.cos(math.radians(theta[t]))
                        y = center.y + r*math.sin(math.radians(theta[t]))
                    else:
                        r = t / 13 + 20
                        x = center.x + r*math.cos(math.radians(theta[t%13]))
                        y = center.y + r*math.sin(math.radians(theta[t%13]))
                    break
                t += 1
        #화염차
        if unit.type_id is UnitTypeId.HELLION:
            for hel in self.bot.helArray:
                if unit.tag == hel: 
                    if t == 0: 
                        r = 23
                        x = center.x + r*math.cos(math.radians(theta[t]))
                        y = center.y + r*math.sin(math.radians(theta[t]))
                    else:
                        r = t / 13 + 23
                        x = center.x + r*math.cos(math.radians(theta[t%13]))
                        y = center.y + r*math.sin(math.radians(theta[t%13]))
                    break
                t += 1
        #탱크
        if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
            for tank in self.bot.tankArray:
                if unit.tag == tank: 
                    if t == 0: 
                        r = 26
                        x = center.x + r*math.cos(math.radians(theta[t]))
                        y = center.y + r*math.sin(math.radians(theta[t]))
                    else:
                        r = t / 13 + 26
                        x = center.x + r*math.cos(math.radians(theta[t%13]))
                        y = center.y + r*math.sin(math.radians(theta[t%13]))
                    break
                t += 1

        target_pos = (Point2((x, y)))

        return target_pos


    def moving(self, unit, actions, target):
        """
        전차의 이동과 변신 담당
        """
        t=0
        #print(len(self.bot.tankArray))
        #print("위치:",self.position_list[t])
        
        for tank in self.bot.tankArray: #이동
            if unit.tag == tank:
                if self.distance(unit.position, self.position_list[t]) < 1: #지정위치도착
                    if unit.type_id == UnitTypeId.SIEGETANK: 
                        order = unit(AbilityId.SIEGEMODE_SIEGEMODE) #변신함
                        actions.append(order)
                        if self.marine_call == 0: self.marine_call = 1 #하나라도 변신하면 해병 와라
                else: #지정위치도착x
                    if unit.type_id == UnitTypeId.SIEGETANKSIEGED: #변신중이면 변신풀기
                        order = unit(AbilityId.UNSIEGE_UNSIEGE) 
                        actions.append(order)
                    elif self.distance(unit.position, self.position_list[t]) < 5 and self.bot.fighting: 
                            #변신은 안했는데 싸우고 있으면 일단 적 때리고, 위치는 너무 나가지 않게
                            actions.append(unit.attack(target))
                            if self.marine_call == 0: self.marine_call = 1
                    else: #변신도 안했고 가까이서 싸우지도 않으면 자기 자리 찾기
                        actions.append(unit.move(self.position_list[t]))
            t+=1
    
    def move_check_action(self, move_check):
        """
        무브체크 변경하면 할 일
        move_check를 그대로 받아씀에 주의
        """
        self.position_list = list()
        self.circle2(self.tank_center[move_check])
        self.move_check = move_check
        self.marine_call = 0
        self.before_marine = self.now_marine #과거 마린 수는 무브체크 변경시마다 기록, 현재 수는 매 스텝마다 기록
        self.move_time = self.bot.time #탱크 이동 시간 기록

    def nuke_action(self, unit, actions):
        """
        핵과의 거리가 가까우면 도망(또는 멈춰있음?)
        """
        distance = unit.position.x - self.bot.nuke_pos.x
        if distance > 0:
            distance = 10 - distance
        else:
            distance = -10 - distance

        if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
            order = unit(AbilityId.UNSIEGE_UNSIEGE) 
            actions.append(order)
        else:
            actions.append(unit.move(Point2((unit.position.x + distance, unit.position.y))))
        
    
    async def step(self):
        #region Description
        actions = list()

        enemy_cc = self.bot.enemy_cc  # 적 시작 위치
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        mule = self.bot.units(UnitTypeId.MULE)
        mule_pos = self.bot.cc.position.towards(enemy_cc.position, -5) #뮬 기본 집결지
        
        #다친 기계 유닛(탱크)
        wounded_units = self.bot.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 1.0 and u.type_id is not UnitTypeId.AUTOTURRET
        )  # 체력이 100% 이하인 유닛 검색, 오토터렛 제외

        #얼마나 많은 유닛이 공격중인가?(적 유닛을 상대로)
        attacking_units = self.bot.combat_units.filter(
            lambda u: u.is_attacking 
            and u.order_target in self.bot.known_enemy_units.tags
        )  # 유닛을 상대로 공격중인 유닛 검색

        self.now_marine = self.bot.combat_units.amount - len(self.bot.tankArray) #현재 해병 수 갱신

        self.position_list = list()
        self.circle2(self.tank_center[0])
        #endregion

        #region Update
        ##-----플래그 변경-----
        #5명 이상 공격중이면 fighting인걸로
        if attacking_units.amount > 5:
            self.bot.fighting = 1
        else: self.bot.fighting = 0
        '''
        ##-----move_check 변화-----
        if self.now_marine < 15 and self.move_check >= 1: 
            self.less_marine = self.bot.time #해병 15 이하인 시간 체크
            if self.bot.time - self.move_time > 10 and self.before_marine >= self.now_marine:
                #10초가 흘렀는데 해병이 늘지 않았으면 바로 후퇴(아니면 25초까지 기다려봄)
                if self.bot.units(UnitTypeId.RAVEN).amount > 0: #밤까마귀 만드느라 생산 막힌 경우 제외
                    self.move_check_action(self.move_check-1)
            elif self.bot.time - self.move_time > 10 and self.bot.time - self.less_marine > 25:
                # 해병이 15이하인게 25초 이상 유지된다면 무브체크 타이밍에 무브체크 감소(밤까마귀)
                self.move_check_action(self.move_check-1)
        
        elif self.bot.combat_units.amount <= 40 and self.move_check == 0 and self.bot.tank_units.amount <= 3: #초기상태
            self.position_list = list()
            self.circle2(self.tank_center[0])
        elif self.bot.combat_units.amount > 40 and self.move_check == 0 and self.bot.tank_units.amount > 3 and self.bot.fighting == 0:
            self.move_check_action(self.move_check+1)
        elif self.bot.combat_units.amount > 40 and self.move_check == 1 and self.bot.tank_units.amount > 3 and self.bot.time - self.move_time > 10 and self.bot.fighting == 0:
            self.move_check_action(self.move_check+1)
        elif self.bot.combat_units.amount > 40 and self.move_check == 2 and self.bot.tank_units.amount > 5 and self.bot.time - self.move_time > 10 and self.bot.fighting == 0:
            self.move_check_action(self.move_check+1)
        elif self.bot.combat_units.amount > 40 and self.move_check == 3 and self.bot.tank_units.amount > 7 and self.bot.time - self.move_time > 10 and self.bot.fighting == 0:
            self.move_check_action(self.move_check+1)
        elif self.move_check ==4 and self.bot.combat_units.amount > self.bot.known_enemy_units.amount and self.bot.time - self.move_time > 10 and self.bot.fighting == 0:
            self.move_check_action(self.move_check+1) 
        
        #탱크 수에 따라 후퇴(해병만 작뜩 모여도 의미가 없는듯)
        elif (self.move_check == 2 or self.move_check == 1) and self.bot.tank_units.amount < 3:
            self.move_check_action(self.move_check-1)
        elif self.move_check ==3 and self.bot.tank_units.amount <= 5:
            self.move_check_action(self.move_check-1)
        elif self.move_check ==4 and self.bot.tank_units.amount <= 7:
            self.move_check_action(self.move_check-1) 
         
        elif (self.bot.tank_units.amount <= 1 or self.bot.combat_units.amount < 5) and self.move_check >= 1: #인원 대폭 줄었으면 초기 위치로
            self.move_check_action(0)'''
        #endregion
        #region Mule
        ##-----Mule 기계 유닛 힐-----
        if self.bot.cc.health_percentage < 0.8: #cc피가 우선
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
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, Point2((wounded_unit.position.x-2, wounded_unit.position.y))))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
        else: #택틱이 뮬인데 치료할거리가 없을경우(치료 끝난 후)
            if mule.amount > 0:
                mule_unit=mule.random
                if self.bot.cc.health_percentage < 1.0: #우선순위의 것들 다 치료했는데 cc가 풀피 아니면 그쪽 힐하러 감
                    actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, self.bot.cc))
                elif self.bot.combat_units.exists:
                    actions.append(mule_unit.move(self.bot.combat_units.center))
                else: actions.append(mule_unit.move(mule_pos))

        #endregion
        
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        self.n = 0
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            if unit in self.bot.combat_units:
                ##-----타겟 설정-----
                enemy_unit = None
                threaten = self.bot.known_enemy_units
                #공중 공격 가능이면 공중 우선 타겟팅(고스트 최우선-핵방어)
                #해병
                if unit.type_id is UnitTypeId.MARINE:
                    enemy_ghost = threaten(UnitTypeId.GHOST)
                    enemy_raven = threaten(UnitTypeId.RAVEN)
                    enemy_bansee = threaten(UnitTypeId.BANSHEE)
                    flying_target = threaten.filter(
                        lambda unit:  unit.is_flying
                    )
                    flying_buff = flying_target.filter(
                        lambda unit:  unit.has_buff(BuffId.RAVENANTIARMORMISSILEARMORREDUCTION)
                    )
                    ##-----타겟 조정-----
                    if threaten.exists :
                        if enemy_ghost.exists and enemy_ghost.closest_to(unit.position).can_be_attacked :
                            enemy_unit = enemy_ghost.closest_to(unit.position)
                        elif enemy_bansee.exists and enemy_bansee.closest_to(unit.position).can_be_attacked:
                            enemy_unit = enemy_bansee.closest_to(unit.position)
                        elif flying_buff.exists:
                            enemy_unit = flying_buff.closest_to(unit)
                        elif flying_target.exists:
                            enemy_unit = flying_target.closest_to(unit)
                        else:
                            enemy_unit = threaten.closest_to(unit.position)
                #바이킹
                if unit.type_id is UnitTypeId.VIKINGFIGHTER:
                    enemy_bansee = threaten(UnitTypeId.BANSHEE)
                    enemy_battle = threaten(UnitTypeId.BATTLECRUISER)
                    flying_target = threaten.filter(
                        lambda unit:  unit.is_flying
                    )
                    flying_buff = flying_target.filter(
                        lambda unit: unit.has_buff(BuffId.RAVENANTIARMORMISSILEARMORREDUCTION)
                    )
                    ##-----타겟 조정-----
                    if threaten.exists :
                        if enemy_battle.exists:
                            enemy_unit = enemy_battle.closest_to(unit.position)
                        elif enemy_bansee.exists and enemy_bansee.closest_to(unit.position).can_be_attacked:
                            enemy_unit = enemy_bansee.closest_to(unit.position)
                        elif flying_buff.exists:
                            enemy_unit = flying_buff.closest_to(unit)
                        elif flying_target.exists:
                            enemy_unit = flying_target.closest_to(unit.position)
                
                #지상만 공격 가능이면 공중은 타겟팅 안함
                if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, UnitTypeId.HELLION):
                    if self.bot.known_enemy_units.exists:
                        walking_target = self.bot.known_enemy_units.filter(
                            lambda u: not u.is_flying
                        )
                        enemy_ghost = self.bot.known_enemy_units(UnitTypeId.GHOST)
                        if enemy_ghost.exists and enemy_ghost.closest_to(unit.position).can_be_attacked:
                            enemy_unit = enemy_ghost.closest_to(unit.position)
                        elif walking_target.exists:
                            enemy_unit = walking_target.closest_to(unit)

                if enemy_unit is None and unit.distance_to(enemy_cc) < 15:
                    target = enemy_cc
                else:
                    target = enemy_unit
                ravens = self.bot.units(UnitTypeId.RAVEN)
                
                #모든 유닛이 근처에 핵 발견했으면 뒤로 도망가는게 최우선
                if self.bot.nuke_alert and unit.distance_to(self.bot.nuke_pos) < 11: 
                    enemy_ghost = self.bot.known_enemy_units(UnitTypeId.GHOST)
                    if ravens.exists and self.bot.run_alert == 0:
                        if self.bot.time - self.bot.nuke_time > 11 :
                            if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                            else: actions.append(unit.move(self.bot.runaway(unit.position,self.bot.nuke_pos,13)))
                        elif enemy_ghost.exists and enemy_ghost.closest_to(unit).can_be_attacked:
                            actions.append(unit.attack(enemy_ghost.closest_to(unit)))
                        #TODO : 나중에 더 세세하게 생각해보기
                    else:
                        if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                            actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                        else: actions.append(unit.move(self.bot.runaway(unit.position,self.bot.nuke_pos,13)))

                ##-----MARINE-----
                elif unit.type_id is UnitTypeId.MARINE:

                    ##-----명령-----
                    if target is not None:
                        actions.append(unit.attack(target))
                        if not unit.has_buff(BuffId.STIMPACK) and unit.distance_to(target) < 15 and threaten.amount > 5:
                                # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                                # '''not unit.has_buff(BuffId.STIMPACK) and''' 여기 주석했음
                            if unit.health_percentage > 0.5:
                                # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                                if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                    # 1초 이전에 스팀팩을 사용한 적이 없음
                                    actions.append(unit(AbilityId.EFFECT_STIM))
                                    self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time
                    #DEFENSE
                    elif self.bot.combat_strategy == 0 : 
                        target_pos = self.defense_circle(unit,self.defense_center) #자신의 대기위치 계산
                        if self.distance(unit.position, target_pos) > 0:
                            actions.append(unit.move(target_pos))
                        #else: actions.append(unit.hold_position)

                        #if self.bot.known_enemy_units.closer_than(10, unit.position).exists:
                            #actions.append(unit.attack(target))

                    #WAIT
                    elif self.bot.combat_strategy == 1: 
                        target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        if self.distance(unit.position, target_pos) > 0:
                            actions.append(unit.move(target_pos))
                    
                    #OFFENSE
                    else:
                        #target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        #if self.distance(unit.position, target_pos) > 0:
                        actions.append(unit.move(self.bot.enemy_cc))


                    ##-----스킬-----
                    

                ##-----HELLION-----
                elif unit.type_id is UnitTypeId.HELLION:

                    if target is not None:
                        position = None

                        if threaten.exists:
                            for u in threaten:
                                if unit.weapon_cooldown == 0:
                                    actions.append(unit.attack(target))
                                elif unit.weapon_cooldown < 18:
                                    if u.target_in_range(unit, 0.5):
                                        actions.append(unit.move(self.bot.runaway(unit.position, u.position,10)))
                                    else:
                                        actions.append(unit.attack(target))
                                else:
                                    if u.target_in_range(unit, 1):
                                        actions.append(unit.move(self.bot.runaway(unit.position, u.position,10)))
                                    else:
                                        actions.append(unit.attack(target))
                        
                        else:
                            if unit.weapon_cooldown == 0:
                                actions.append(unit.attack(target))
                            elif unit.weapon_cooldown < 18:
                                if target.target_in_range(unit, 0.5):
                                    actions.append(unit.move(self.bot.runaway(unit.position, target.position,10)))
                                else:
                                    actions.append(unit.attack(target))
                            else:
                                if u.target_in_range(unit, 1):
                                    actions.append(unit.move(self.bot.runaway(unit.position, u.position,10)))
                                else:
                                    actions.append(unit.attack(target))

                        '''
                        if unit.distance_to(target) <=4:
                            position = self.bot.run
                            away(unit.position, target.position,10)
                            actions.append(unit.move(position))
                        el
                        elif unit.weapon_cooldown < 18:

                            position = self.bot.runaway(unit.position, target.position,10)
                            actions.append(unit.move(position))
                    '''
                    elif self.bot.cloak_units.exists:
                        c = self.bot.cloak_units.closer_than(7,unit)
                        if c.exists:
                            actions.append(unit.move(self.bot.runaway(unit.position,c.position,10)))
                    #DEFENSE
                    elif self.bot.combat_strategy == 0 : 
                        target_pos = self.defense_circle(unit,self.defense_center) #자신의 대기위치 계산
                        if self.distance(unit.position, target_pos) > 0:
                            actions.append(unit.move(target_pos))                  
                    #WAIT
                    elif self.bot.combat_strategy == 1: 
                        target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        if self.distance(unit.position, target_pos) > 0:
                            actions.append(unit.move(target_pos))
                    #OFFENSE
                    else:
                        #target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        #if self.distance(unit.position, target_pos) > 0:
                        actions.append(unit.move(self.bot.enemy_cc))
                    
                    
                ##-----TANK-----
                elif unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                    ##-----명령-----
                    if target is not None:
                        actions.append(unit.attack(target))
                    #DEFENSE
                    elif self.bot.combat_strategy == 0 : 
                        target_pos = self.defense_circle(unit,self.defense_center) #자신의 대기위치 계산
                        if self.distance(unit.position, target_pos) == 0: #도착
                            if unit.type_id is UnitTypeId.SIEGETANK:
                                actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE)) #변신
                        else:
                            if unit.type_id is UnitTypeId.SIEGETANKSIEGED: 
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE)) #변신 풀기
                            else: actions.append(unit.move(target_pos))                  
                    #WAIT
                    elif self.bot.combat_strategy == 1: 
                        target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        if self.distance(unit.position, self.bot.enemy_cc) == 0: #도착
                            if unit.type_id is UnitTypeId.SIEGETANK:
                                actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE)) #변신
                        else:
                            if unit.type_id is UnitTypeId.SIEGETANKSIEGED: 
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE)) #변신 풀기
                            else: actions.append(unit.move(target_pos))
                    #OFFENSE
                    else:
                        #target_pos = self.defense_circle(unit,self.bot.enemy_cc) #자신의 대기위치 계산
                        #if self.distance(unit.position, target_pos) > 0:
                        if unit.type_id is UnitTypeId.SIEGETANKSIEGED: 
                            actions.append(unit(AbilityId.UNSIEGE_UNSIEGE)) #변신 풀기
                        else: actions.append(unit.move(self.bot.enemy_cc))


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
     

class TrainManager(object):
    """
    CombatManager의 생산 유닛을 결정
    유령과 밤까마귀는 각 매니저에서 생산 결정
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    def next_unit(self):
        if self.bot.vespene > 100:
            if len(self.bot.tankArray) <= 20:
                next_unit = UnitTypeId.SIEGETANK
            else:
                next_unit = UnitTypeId.BATTLECRUISER
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
        유닛 타입을 보고 부대별 array에 유닛 배치 및 갱신
        """
        #print("싸우고 있나?: ", self.bot.fighting)
        units_tag = self.bot.units.tags #전체유닛
        
        #and연산으로 살아있는 유닛으로만 구성
        self.bot.combatArray = self.bot.combatArray & units_tag 
        self.bot.reconArray = self.bot.reconArray & units_tag
        self.bot.nukeArray = self.bot.nukeArray & units_tag

        #탱크 어레이에 배정되었다가 죽은 탱크는 None으로
        tank_tag = self.bot.units(UnitTypeId.SIEGETANKSIEGED).tags | self.bot.units(UnitTypeId.SIEGETANK).tags
        j = 0
        for tag in self.bot.tankArray:
            i = 1
            
            for tag1 in tank_tag:
                if tag == tag1:
                    i = 0
            if i:
                self.bot.tankArray[j]=None
            
            j += 1

        #마린 어레이에 배정되었다가 죽은 마린은 None
        marine_tag = self.bot.units(UnitTypeId.MARINE).tags
        a = 0
        for tag in self.bot.marineArray:
            b = 1
            
            for tag1 in marine_tag:
                if tag == tag1:
                    b = 0
            if b:
                self.bot.marineArray[a]=None
            
            a += 1

        #헬리온 어레이에 배정되었다가 죽은 마린은 None
        hel_tag = self.bot.units(UnitTypeId.HELLION).tags
        n = 0
        for tag in self.bot.helArray:
            m = 1
            
            for tag1 in hel_tag:
                if tag == tag1:
                    m = 0
            if m:
                self.bot.helArray[n]=None
            
            n += 1


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
            elif unit.type_id is UnitTypeId.VIKINGFIGHTER: #바이킹 컴뱃
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.HELLION: #화염차 컴뱃
                self.bot.combatArray.add(unit.tag)
                #헬리온에 None인 곳이 있으면 거기 먼저 배치
                m = 1
                n = 0
                for tag in self.bot.helArray:
                    if tag == None:
                        self.bot.helArray[n] = unit.tag
                        m = 0
                        break
                    n += 1
                if m: 
                    self.bot.helArray.append(unit.tag)
            elif unit.type_id in (UnitTypeId.SIEGETANKSIEGED,  UnitTypeId.SIEGETANK): #탱크(변신)는 컴뱃
                self.bot.combatArray.add(unit.tag)
                #탱크는 탱크에도 넣는데 None인 곳이 있으면 거기 먼저 배치
                j = 1
                i = 0
                for tag in self.bot.tankArray:
                    if tag == None:
                        self.bot.tankArray[i] = unit.tag
                        j = 0
                        break
                    i += 1
                if j: 
                    self.bot.tankArray.append(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)
                #마린에 None인 곳이 있으면 거기 먼저 배치
                a = 1
                b = 0
                for tag in self.bot.marineArray:
                    if tag == None:
                        self.bot.marineArray[b] = unit.tag
                        a = 0
                        break
                    b += 1
                if a: 
                    self.bot.marineArray.append(unit.tag)
                
    '''
    def reassign(self):
        """
        이미 배정된 유닛을 다시 배정
        있을리 없는 유닛타입을 옮겨줌-유닛타입에 따라 배치 하고 나서 안씀
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
                self.bot.combatArray.add(unit.tag)'''


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    공격 타이밍은 강화학습을 통해 학습
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        
        self.host_name = host_name
        """
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc() """

        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.assign_manager = AssignManager(self)
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        self.train_manager = TrainManager(self)

        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()
        self.tankArray = list()
        self.marineArray = list()
        self.helArray = list()
        self.vikArray = list()

        self.nuke_reward = 0 
        self.nuke_strategy= 2
        self.combat_strategy = 0 #0:Defense, 1: WAIT, 2:Offense(무브체크 대신)

        # 정찰부대에서 사용하는 플래그
        self.threaten=list()
        self.enemy_alert=0
        self.die_alert=0 # 레이븐 죽었을때
        self.is_ghost=0 # 적이 유령인지 -> 안씀???
        self.last_pos=(0,0) # 레이븐 실시간 위치

        # 핵 부대에서 사용하는 플래그
        self.die_count=0
        self.ghost_ready = False

        # assign에서 사용하는 플래그(갱신은 컴뱃, 레콘에서 함)
        self.fighting = 0 #현재 컴뱃에서 싸우고 있는가? 0:No, 1:Yes
        self.have_to_go = 0 #레콘으로 옮겨야하는가? 0:No, 1:일부만, 2:전체

        self.nuke_alert = False
        self.command_nuke = False # 핵이 사령부 주위에 떨어지는지 판단
        self.nuke_time = 0
        self.nuke_pos = None
        self.nuketime_flag = 0
        self.is_nuke = 0
        self.run_alert = 0
        #self.is_raven = 0
        
    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        if self.start_location.x > 40: #blue
            self.enemy_cc = Point2(Point2((32.5,31.5)))
        else: #red
            self.enemy_cc = Point2(Point2((95.5,31.5)))
        
        self.enemy_is_flying = False #적이 공중 주력이면 True됨

        self.nuke_position =  self.enemy_cc #핵 목표 위치

        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.evoked = dict() #생산명령 시간 체크

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()
        self.nuke_manager.reset()
        
        self.assign_manager.assign()

        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"

    def runaway(self, u, t, dis):
        # 상대위치와 내 위치로 직선방정식 구하기
        # 상대위치에서 내 위치 방향으로 7 이상 떨어지게 하기 (상대와 내가 5거리, 그러면 내 점과 2 떨어진 점 중 직선위에 있는, 상대점과 7 떨어진 점)
 
        if u.x == t.x :
            if u.y < t.y:
                position = Point2((u.x,t.y-dis))
            else:
                position = Point2((u.x,t.y + dis))
        else:
            a = (t.y-u.y)/(t.x-u.x) # 기울기
            d = dis - math.sqrt((u.x-t.x)**2+(u.y-t.y)**2)
            
            p = d*d/(a**2+1)
            q = math.sqrt(p)

            if u.x < t.x : #유닛이 왼쪽이면 왼쪽으로 도망
                x = u.x - q
            else:
                x = u.x + q
            y = a*(x - u.x) + u.y
            position = Point2((x,y))

 
        return position
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌
        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록

        #전체 전진 기준
        if self.combat_strategy == 0 and len(self.tankArray) >=3 and self.units(UnitTypeId.VIKINGFIGHTER).amount >=4 and self.combat_units.amount > 60:
            self.combat_strategy = 1
        if self.combat_strategy == 1 and len(self.tankArray) + len(self.vikArray) >= 15 and self.combat_units.amount > 60:
            self.combat_strategy = 2

        self.flying_enemy = self.known_enemy_units.filter(
            lambda u: u.type_id in (UnitTypeId.BANSHEE, UnitTypeId.BATTLECRUISER, UnitTypeId.VIKINGFIGHTER)
        ) #적 공중 유닛(공격용)
        self.walking_enemy =  self.known_enemy_units.filter(
            lambda u: not u.is_flying
        ) #적 지상 유닛

        #적이 공중전 타입이라는 걸 체크 -- 수정 필요
        if self.flying_enemy.amount >= self.walking_enemy.amount / 2 or self.known_enemy_units(UnitTypeId.BATTLECRUISER).exists:
            self.enemy_is_flying = True
        '''
        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경
            before = self.nuke_strategy
            self.nuke_strategy = self.set_strategy()
            if self.units(UnitTypeId.GHOST).amount > 0:
                if before != self.nuke_strategy:
                    self.nuke_reward -= 0.001
                else :
                    self.nuke_reward += 0.001
            #nuke reward 초기화
            self.nuke_reward = 0
        ''' # 고스트만 강화학습하는 가정 하에 해봄

        '''
        if self.time - self.last_step_time >= self.step_interval and self.units(UnitTypeId.GHOST).amount == 0:
            self.nuke_strategy = self.set_strategy()
        '''
        #self.assign_manager.reassign() #이상하게 배치된 경우 있으면 제배치
        #self.last_step_time = self.time

        self.cloak_units = self.known_enemy_units.filter(
            lambda u: u.type_id in (UnitTypeId.BANSHEE, UnitTypeId.GHOST) and u.cloak == 1
        ) #적 은폐 유닛 탐지
      
        self.nuke_alert = False
        self.is_nuke=1
        for effect in self.state.effects:
            if effect.id == EffectId.NUKEPERSISTENT:
                self.is_nuke = 0
                self.nuke_alert = True
                nuke_list = list(effect.positions)
                self.nuke_pos = nuke_list[0]
                if self.nuketime_flag == 0:
                    self.nuketime_flag = 1
                    self.nuke_time = self.time
                #사령부 주변에 핵 잡히면 밤까마귀가 처치하도록
                if self.units(UnitTypeId.COMMANDCENTER).exists and self.units(UnitTypeId.COMMANDCENTER).first.distance_to(self.nuke_pos) < 8:
                    self.command_nuke = True
                else:
                    self.command_nuke = False

        if self.nuke_alert and self.nuketime_flag == 1:
            self.nuketime_flag = 2

        # 핵이 완전히 사라짐
        if self.time - self.nuke_time >= 15 or self.is_nuke:
            self.nuke_alert = False
            self.command_nuke = False
            self.nuketime_flag = 0
            self.run_alert = 0

        actions += await self.train_action() #생산
        self.assign_manager.assign()

        #-----유닛츠 배치-----
        self.combat_units = self.units.filter(
            lambda unit: unit.tag in self.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, UnitTypeId.HELLION, UnitTypeId.VIKINGFIGHTER]
        )
        self.nuke_units = self.units.tags_in(self.nukeArray)
        self.tank_units = self.combat_units.filter(
            lambda unit:  unit.type_id in [UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        self.marine_units = self.combat_units.filter(
            lambda u: u.type_id is UnitTypeId.MARINE
        )
        self.hel_units = self.combat_units.filter(
            lambda u: u.type_id is UnitTypeId.HELLION
        )
        
        actions += await self.recon_manager.step() 
        actions += await self.combat_manager.step()   
        actions += await self.nuke_manager.step()
        #self.assign_manager.assign() #생산을 리콘과 누크에서 해줘서 추가한거였는데 이제 아니니까 주석
        
        ## -----명령 실행-----
        await self.do_actions(actions)


    async def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()
        #next_unit = self.train_manager.next_unit()
        next_unit = None
        ravens = self.units(UnitTypeId.RAVEN)
        # 1순위: 핵이나 은폐 있을때 밤까마귀
        # 2순위: 유령(시간됐을떄)
        # 3순위: 핵
        # 4순위: 넥스트 유닛(마린, 탱크)

        cc_abilities = await self.get_available_abilities(self.cc)
        #핵 우선생산
        # 이미 핵이 있으면 생산X
        if AbilityId.BUILD_NUKE not in cc_abilities and self.ghost_ready:
            self.ghost_ready = False

        #은폐 또는 핵 감지했을 떄 레이븐 없으면 레이븐 먼저
        if (self.nuke_alert and self.command_nuke and self.time - self.nuke_time < 9 and ravens.closest_to(self.nuke_pos).distance_to(self.nuke_pos) > 25) or (self.cloak_units.amount > 0 and not ravens.exists) :
            # 시간넉넉하면 밤까마귀 생성해서 막기
            # train_action에서 플래그 보고 자원 아껴야함
                next_unit = UnitTypeId.RAVEN


        else: 
            #핵이나 은폐있는데 베스핀 없음 -> 밤까마귀 존버
            if (self.nuke_alert or self.cloak_units.exists) and not self.units(UnitTypeId.RAVEN).exists and self.vespene <= 200:
                next_unit = None
                pass 
            #초기 집결 중(밤까마귀 생산 필요x)
            elif self.combat_strategy == 0:
                #밤까마귀도 있는데 은폐유닛이 남아있어->바이킹
                if self.units(UnitTypeId.RAVEN).exists and self.cloak_units.exists and self.vespene > 50:
                    next_unit = UnitTypeId.VIKINGFIGHTER
                #베스핀 남으면 우선 탱크 3대까지(베스핀 여유 두어야)
                elif self.known_enemy_units.amount >= 5 and self.tank_units.amount < 4:
                    next_unit = UnitTypeId.SIEGETANK
                elif self.vespene > 250 and self.tank_units.amount <= self.units(UnitTypeId.VIKINGFIGHTER).amount and self.tank_units.amount < 4:
                    next_unit = UnitTypeId.SIEGETANK
                #바이킹은 4대까지(베스핀 여유두고)
                elif self.vespene > 250 and self.tank_units.amount >= self.units(UnitTypeId.VIKINGFIGHTER).amount and self.tank_units.amount < 5:
                    next_unit = UnitTypeId.VIKINGFIGHTER
                #바이킹 4, 탱크 3 다 완성되면 탱크와 바이킹 수 똑같이 생산되도록
                elif self.vespene > 250 and self.tank_units.amount <= self.units(UnitTypeId.VIKINGFIGHTER).amount:
                    next_unit = UnitTypeId.SIEGETANK
                elif self.vespene > 250 and self.tank_units.amount >= self.units(UnitTypeId.VIKINGFIGHTER).amount:
                    next_unit = UnitTypeId.VIKINGFIGHTER
                #해병과 화염차도 미네랄 여유 두고 생산(2:1비율)
                elif self.minerals >= 200:
                    if self.units(UnitTypeId.HELLION).amount < self.units(UnitTypeId.MARINE).amount / 2:
                        next_unit = UnitTypeId.HELLION
                    else: next_unit = UnitTypeId.MARINE
            #초기 집결 후 전진 중(밤까마귀 생산 필요x)
            else:
                #고스트가 없고 200초 지남, die_count가 2 이하 -> 유령 생산
                if self.units(UnitTypeId.GHOST).amount == 0 and self.die_count < 2 and self.time >= 200:
                    next_unit = UnitTypeId.GHOST
                    self.ghost_ready = True # 핵 항상 생산
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                elif self.ghost_ready and AbilityId.BUILD_NUKE in cc_abilities and self.units(UnitTypeId.GHOST).exists:
                    actions.append(self.cc(AbilityId.BUILD_NUKE))
                    next_unit = None
                    self.ghost_ready = False 
                #누크 관련 우선 생산하고, 누크 조건에 걸리지 않으면
                else:
                    #적이 지상전 타입이면
                    if self.enemy_is_flying == False:
                        if self.vespene > 100: #베스핀 되는대로 탱크
                            next_unit = UnitTypeId.SIEGETANK
                        elif self.units(UnitTypeId.HELLION).amount < self.units(UnitTypeId.MARINE).amount / 2:
                            next_unit = UnitTypeId.HELLION
                        else: next_unit = UnitTypeId.MARINE
                    #적이 공중전 타입이면
                    else:
                        if self.vespene > 50: #베스핀 되는대로 바이킹
                            next_unit = UnitTypeId.VIKINGFIGHTER
                        elif self.units(UnitTypeId.HELLION).amount < self.units(UnitTypeId.MARINE).amount / 2:
                            next_unit = UnitTypeId.HELLION
                        else: next_unit = UnitTypeId.MARINE
        
        if next_unit != None and self.can_afford(next_unit) and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
            actions.append(self.cc.train(next_unit))
            self.evoked[(self.cc.tag, 'train')] = self.time

        return actions

    '''
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
        return nuke_strategy.value'''

    '''
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
    '''

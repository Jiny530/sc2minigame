
__author__ = '이다영, 박혜진'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId

from enum import Enum
from random import *


# 주 부대에 속한 유닛 타입
ARMY_TYPES = (UnitTypeId.MARINE, UnitTypeId.MARAUDER, 
    UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.MEDIVAC)
    
class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        
    # 마지막 명령이 발행된지 10초가 넘었으면 리워드 -1?
    # 택틱마다 일정시간 지나도 명령 발행 안되면 리워드 -1 

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        
        if ghosts.amount == 0:
            if AbilityId.BARRACKSTRAIN_GHOST in cc_abilities:
                # 고스트가 하나도 없으면 고스트 훈련
                print(cc_abilities)
                actions.append(cc.train(UnitTypeId.GHOST))

        elif ghosts.amount > 0:
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(cc(AbilityId.BUILD_NUKE))

            ghost_abilities = await self.bot.get_available_abilities(ghosts.first)
            if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghosts.first.is_idle:
                # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                actions.append(ghosts.first(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.enemy_cc))

        return actions


class ReconManager(object):
    """
    정찰부대 운용을 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.perimeter_radious = 10

    def reset(self):
        self.perimeter_radious = 10

    # def position(self):
        
    async def step(self):
        actions = list()

        ravens = self.bot.units(UnitTypeId.RAVEN)

        for unit in ravens: 
            # 근처에 적들이 있는지 파악
            threaten = self.bot.known_enemy_units.closer_than(
                    self.perimeter_radious, unit.position)

            if unit.health_percentage > 0.8 and unit.energy >= 50:
                print("유닛오더? ",unit.orders)
                if threaten.amount > 0: # 근처에 적이 하나라도 있으면
                    alert = 1
                    if unit.orders and unit.orders[0].ability.id != AbilityId.RAVENBUILD_AUTOTURRET:
                        closest_threat = threaten.closest_to(unit.position)
                        pos = unit.position.towards(closest_threat.position, 5)
                        pos = await self.bot.find_placement(
                            UnitTypeId.AUTOTURRET, pos)
                        order = unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos)
                        actions.append(order)
                '''else:
                    if unit.distance_to(self.target) > 5:
                        order = unit.move(self.target)
                        actions.append(order)'''
            else:
                if unit.distance_to(self.bot.start_location) > 5:
                    order = unit.move(self.bot.start_location)
                    actions.append(order)
        return actions
    

class Combat_Team_Manager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):

        self.evoked = dict()

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        combat_units = self.bot.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        wounded_units = self.bot.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치

        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            
            ##-----타겟 설정-----
            enemy_unit = self.bot.enemy_start_locations[0]
            if self.bot.known_enemy_units.exists:
                enemy_unit = self.bot.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            ##-----유닛 스킬-----

            ##-----전투 유닛 전체-----
            if unit.type_id is not UnitTypeId.MEDIVAC:
                #print(combat_units.amount)
                if combat_units.amount > 20 and self.bot.units.of_type(UnitTypeId.SIEGETANK).amount > 1:
                    # 전투가능한 유닛 수가 15을 넘으면 + 탱크 2대 이상 적 본진으로 공격
                    actions.append(unit.attack(target))
                    use_stimpack = True
                else:
                    # 적 사령부 방향에 유닛 집결
                    target = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
                    actions.append(unit.attack(target))
                    use_stimpack = False

                ##-----해병과 불곰-----
                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if use_stimpack and unit.distance_to(target) < 15:
                        # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time

                ##-----탱크-----
                if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                    if self.bot.known_enemy_units.exists and unit.distance_to(target) < 15 : 
                        if unit.type_id == UnitTypeId.SIEGETANK:
                            order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                            actions.append(order)
                        
                        else:
                            #특정 유닛 타입 골라서 타겟팅하는거 에러있음 
                            #target =  target.type_id([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
                            order = unit(AbilityId.UNSIEGE_UNSIEGE)
                            actions.append(order)
                        
                        actions.append(unit.attack(target))

            ##-----비전투 유닛==메디박-----
            if unit.type_id is UnitTypeId.MEDIVAC:
                ##힐을 가장 우선시
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                    #print(wounded_unit.name, "을 회복중")
                else: 
                    actions.append(unit.move(combat_units.center))
                    #print("대기중")

        # -----유닛 명령 생성 끝-----
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


class Tactics(Enum):
    ATTACK = 0
    DEFENSE = 1
    NUKE=2
    RECON=3


class RatioManager(object):
    """
    tactic에 따라 생성할 유닛 비율 결정하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai

    def next_unit_select(self):
        """
        tactic에 따라 다음에 생산할 유닛 return
        """
        
        if self.bot.tactics == Tactics.ATTACK:
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 25,
                UnitTypeId.MARAUDER: 15,
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 3,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 3,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN: 0,
                UnitTypeId.BATTLECRUISER: 0,
            }
            self.evoked = dict()

        elif self.bot.tactics == Tactics.DEFENSE:
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 0,
                UnitTypeId.MARAUDER: 0,
                UnitTypeId.REAPER: 3,
                UnitTypeId.GHOST: 5,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 5,
                UnitTypeId.RAVEN: 5,
                UnitTypeId.BATTLECRUISER: 0,
            }
            self.evoked = dict()

        # -----부족한 유닛 숫자 계산-----
        unit_counts = dict()
        for unit in self.bot.units:
            unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
        
        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련

        return next_unit


class AssignManager(object):
    """
    유닛을 부대에 배치하는 매니저
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self, manager):

        units_tag = self.bot.units.tags #전체유닛
        
        #tactic이 combat이면 combatarray에 주고 나머지엔 combat에서 쓰는게 제외하고 할당 가능?ㅇㅇ
        units_tag = units_tag - self.bot.combatArray - self.bot.reconArray - self.bot.nukeArray
        
        if self.bot.tactics == Tactics.ATTACK or  self.bot.tactics == Tactics.DEFENSE:
            self.bot.combatArray = self.bot.combatArray | units_tag
        elif self.bot.tactics == Tactics.RECON:
            self.bot.reconArray = self.bot.reconArray | units_tag
        elif self.bot.tactics == Tactics.NUKE:
            self.bot.nukeArray = self.bot.nukeArray | units_tag


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    개별 전투 유닛이 적사령부에 바로 공격하는 대신, 15이 모일 때까지 대기
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.step_manager = StepManager(self)
        self.combat_team_manager = Combat_Team_Manager(self)
        self.assign_manager = AssignManager(self)
        self.tactics = Tactics.ATTACK #Tactic은 ATTACK으로 초기화
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        self.ratio_manager = RatioManager(self)
        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()
        

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_manager.reset()
        self.combat_team_manager.reset()
        self.assign_manager.reset()

        self.assign_manager.assign(self.combat_team_manager)

        self.tatics = Tactics.ATTACK #초기 tactic은 ATTACK으로 초기화
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌

        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록
        ccs = self.units(UnitTypeId.COMMANDCENTER).idle  # 전체 유닛에서 사령부 검색

        if self.step_manager.step % 2 == 0:
            
            # -----사령부 명령 생성-----
            if ccs.exists:  # 사령부가 하나이상 존재할 경우
                cc = ccs.first  # 첫번째 사령부 선택
                next_unit = self.ratio_manager.next_unit_select() #RatioManager에서 받아옴
                if self.can_afford(next_unit) and self.time - self.ratio_manager.evoked.get((cc.tag, 'train'), 0) > 1.0:
                    # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                    actions.append(cc.train(next_unit))
                    self.ratio_manager.evoked[(cc.tag, 'train')] = self.time
                    self.assign_manager.assign(self.combat_team_manager)

            
        
        # -----전략 변경 -----
        if self.step_manager.step % 30 == 0:
            i = randint(0, 3) #일단 랜덤으로 변경
            self.tactics = Tactics(i)
            #print(self.tactics)
            
        
            

        actions += await self.combat_team_manager.step()               


        ## -----명령 실행-----
        await self.do_actions(actions)

'''
    def on_end(self, game_result):
        for tag in self.combatArray:
            print("111전투 : ", tag)
        print("-----")
        for tag in self.reconArray:
            print("2222222정찰 : ", tag)
        print("-----")
        for tag in self.nukeArray:
            print("핵 : ", tag)
        print("-----")'''




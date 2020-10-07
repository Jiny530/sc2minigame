
__author__ = '이다영'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId

# 주 부대에 속한 유닛 타입
ARMY_TYPES = (UnitTypeId.MARINE, UnitTypeId.MARAUDER, 
    UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.MEDIVAC)

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


class Attack_Team_Manager(object):
    """
    공격부대 컨트롤
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):

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

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        ccs = self.bot.units(UnitTypeId.COMMANDCENTER).idle  # 전체 유닛에서 사령부 검색
        combat_units = self.bot.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        wounded_units = self.bot.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치

        # -----부족한 유닛 숫자 계산-----
        unit_counts = dict()
        for unit in self.bot.units:
            unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
        
        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        #
        # -----사령부 명령 생성-----
        ## 비율 맞춰서 생산
        #
        if ccs.exists:  # 사령부가 하나이상 존재할 경우
            cc = ccs.first  # 첫번째 사령부 선택
            next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
            if self.bot.can_afford(next_unit) and self.bot.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(cc.train(next_unit))
                self.evoked[(cc.tag, 'train')] = self.bot.time

        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            
            ##-----타겟 설정-----
            enemy_unit = self.bot.enemy_start_locations[0]
            if self.bot.known_enemy_units.exists:
                #enemy_unit = self.known_enemy_units.health_least() #가장 체력이 적은 유닛???
                #print(enemy_unit.health)
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
                        #print("탱크가 적 인식, 적과의 거리:", unit.distance_to(target))
                        if unit.type_id == UnitTypeId.SIEGETANK:
                            order = unit(AbilityId.SIEGEMODE_SIEGEMODE, unit)
                            actions.append(order)
                            #print("탱크 박기 시킴")
                        
                        else:
                            target =  enemy_unit.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
                            actions.append(unit.attack(target))
                            #print("공격 시킴") ## 왜 아예 시즈모드로 변하지를 않는거???

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
    
    
class Defense_Team_Manager():
    """
    수비부대 컨트롤
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai

    def reset(self):

        self.target_unit_counts = {
            UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
            UnitTypeId.MARINE: 10,
            UnitTypeId.MARAUDER: 5,
            UnitTypeId.REAPER: 0,
            UnitTypeId.GHOST: 0,
            UnitTypeId.HELLION: 0,
            UnitTypeId.SIEGETANK: 2,
            UnitTypeId.THOR: 0,
            UnitTypeId.MEDIVAC: 0,
            UnitTypeId.VIKINGFIGHTER: 0,
            UnitTypeId.BANSHEE: 0,
            UnitTypeId.RAVEN: 0,
            UnitTypeId.BATTLECRUISER: 0,
        }
        self.evoked = dict()

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        ccs = self.bot.units(UnitTypeId.COMMANDCENTER).idle  # 전체 유닛에서 사령부 검색
        combat_units = self.bot.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치
        gathering_position = self.bot.start_location + 0.15 * (enemy_cc.position - self.bot.start_location)

        # -----부족한 유닛 숫자 계산-----
        unit_counts = dict()
        for unit in self.bot.units:
            unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
        
        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        #
        # -----사령부 명령 생성-----
        ## 비율 맞춰서 생산
        #
        if ccs.exists:  # 사령부가 하나이상 존재할 경우
            if combat_units.amount < 21:  ## 수비부대의 총 유닛은 20명을 넘지 않게
                cc = ccs.first  # 첫번째 사령부 선택
                next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
                if self.bot.can_afford(next_unit) and self.bot.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
                    # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                    actions.append(cc.train(next_unit))
                    self.evoked[(cc.tag, 'train')] = self.bot.time

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
                if unit.distance_to(gathering_position) > 10:  ## cc---gathering--(10)--unit-----enemy_cc 인경우 처리
                    #현재위치가 집결장소보다 멀면 돌아오기(여유롭게 10정도 차이는 봐줌)
                    target = gathering_position
                    actions.append(unit.attack(target))
                    use_stimpack = False
                elif unit.distance_to(target) < 15:
                    # 적과의 거리가 15 미만이면, 즉 적이 근접해오면 공격
                    actions.append(unit.attack(target))
                    use_stimpack = True                    
                else:
                    # 적 사령부 방향에 유닛 집결(첫 조건(거리10)에 해당하지 않아도 집결지로 이동)
                    target = gathering_position
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
                        #print("탱크가 적 인식, 적과의 거리:", unit.distance_to(target))
                        if unit.type_id == UnitTypeId.SIEGETANK:
                            order = unit(AbilityId.SIEGEMODE_SIEGEMODE, unit)
                            actions.append(order)
                            #print("탱크 박기 시킴")
                        
                        else:
                            target =  enemy_unit.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER])
                            actions.append(unit.attack(target))
                            #print("공격 시킴") ## 왜 아예 시즈모드로 변하지를 않는거???

        # -----유닛 명령 생성 끝-----
        return actions


class AssignManager(object):
    """
    유닛을 부대에 배치하는 매니저
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self, manager):

        units = self.bot.units

        if manager is Attack_Team_Manager:
            units = self.bot.units.of_type(ARMY_TYPES).owned
            unit_tags = units.tags

            # 새 유닛 요청            
            #new_units = self.bot.units & list()

            # 다른 매니저가 사용 중인 유닛 제외
            unit_tags = unit_tags - self.bot.defense_team_manager.unit_tags
            manager.unit_tags = unit_tags

        elif manager is Defense_Team_Manager:
            units = self.bot.units.of_type(ARMY_TYPES).owned
            unit_tags = units.tags

            # 새 유닛 요청            
            #new_units = self.bot.units & list()

            # 다른 매니저가 사용 중인 유닛 제외
            unit_tags = unit_tags - self.bot.attack_team_manager.unit_tags
            manager.unit_tags = unit_tags


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    개별 전투 유닛이 적사령부에 바로 공격하는 대신, 15이 모일 때까지 대기
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.step_manager = StepManager(self)
        self.attack_team_manager = Attack_Team_Manager(self)
        self.defense_team_manager = Defense_Team_Manager(self)
        self.assign_manager = AssignManager(self)

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_manager.reset()
        self.attack_team_manager.reset()
        self.defense_team_manager.reset()
        self.assign_manager.reset()

        self.assign_manager.assign(self.attack_team_manager) #처음 유닛 할당은 공격 부대부터


    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌

        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        if self.step_manager.step % 2 == 0:
            self.assign_manager.assign(self.attack_team_manager)
            #self.assign_manager.assign(self.defense_team_manager)

        # print(self.state.score.total_damage_dealt_life)
        actions = list() # 이번 step에 실행할 액션 목록

        actions += await self.attack_team_manager.step()
        actions += await self.defense_team_manager.step()              


        ## -----명령 실행-----
        await self.do_actions(actions)






##가까운 적을 타겟으로 잡는 코드
"""
    for unit in self.units.not_structure:
        near_unit = self.start_location
        if self.known_own_units.exists:
            near_unit = self.known_own_units.closest_to(unit)  # 가장 가까운 아군 유닛
                    
    if near_unit.is_attacking: # 가까운 공격중인 아군 유닛이 있다면 아군 유닛의 타깃을 목표로 설정
            target=near_unit.order_target
    else:
        # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
        if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
            target = enemy_cc
        else:
            target = enemy_unit
"""
##의료선에 탑승하는 조건문 코드
"""
##힐 안할때 태우고 내리기(태우고 내릴 아군이 있나 체크)
    elif combat_units.exists:
        combat_unit = combat_units.closest_to(unit)
        '''가장 가까운 유닛이 공격중인가를 기준으로 행동해서
        탑승했던 유닛이 내리면 해당 유닛이 기준으로 변경됨
        여기서 계속 태웠다가 내렸다가를 반복하는 것 같음'''
                    
        '''명령을 내리고 실행까지 await로 기다리는 방식도 고려해봐야
        if(유닛 몇개 이상이면):  
            for i in self.units(combat_units).idle:  
                await self.do(i.attack(target)) 꼴'''  

        '''combat_unit은 기본적으로 Ability.Attack을 받고 시작함
        그래서 싸우는 중인지 아닌지는 is_Attacking으로 구분이 안됨''' 
        ##돌격시작하면+누가 타 있다면 내리기
        if (combat_units.amount >= unit.cargo_used+15) and unit.has_cargo:
            if self.time - self.evoked.get((unit.tag, AbilityId.UNLOADALLAT_MEDIVAC), 0) > 1.0:
                actions.append(unit(AbilityId.UNLOADALLAT_MEDIVAC, unit.position))
                self.evoked[(unit.tag, AbilityId.UNLOADALLAT_MEDIVAC)] = self.time
                print("내림, 위치: ", unit.position)
                print("남은자리:",unit.cargo_left,", 탑승중인 유닛 있는가?:",unit.has_cargo)
        ##근방 유닛이 공격중이 아니면+빈자리가 있으면 유닛 들이기(탑승)
        elif not combat_unit.is_attacking and unit.cargo_left >= 1.0:
            if self.time - self.evoked.get((unit.tag, AbilityId.LOAD_MEDIVAC), 0) > 1.0:
                actions.append(unit(AbilityId.LOAD_MEDIVAC, combat_unit))
                self.evoked[(unit.tag, AbilityId.LOAD_MEDIVAC)] = self.time
                print("남은자리:",unit.cargo_left)
                print(combat_unit.name," (공격중:",combat_unit.is_attacking,") 탑승")
    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
    ##다른 아군이 아예 없을 때 대기
"""
                






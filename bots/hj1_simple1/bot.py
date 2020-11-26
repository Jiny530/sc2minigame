
__author__ = '박혜진'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId

class Tactics(object):
    RECON = 0
    nuke = 1

class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        

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

class Bot(sc2.BotAI):
    """
    해병 5, 의료선 1 빌드오더를 계속 실행하는 봇
    해병은 적 사령부와 유닛중 가까운 목표를 향해 각자 이동
    적 유닛또는 사령부까지 거리가 15미만이 될 경우 스팀팩 사용
    스팀팩은 체력이 50% 이상일 때만 사용가능
    의료선은 가장 가까운 체력이 100% 미만인 해병을 치료함
    """
    a=0

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_order = list() # 생산할 유닛 목록
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.target_unit_counts = {
            UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
            UnitTypeId.MARINE: 15,
            UnitTypeId.MARAUDER: 2,
            UnitTypeId.REAPER: 0,
            UnitTypeId.GHOST: 0,
            UnitTypeId.HELLION: 5,
            UnitTypeId.SIEGETANK: 2,
            UnitTypeId.THOR: 1,
            UnitTypeId.MEDIVAC: 1,
            UnitTypeId.VIKINGFIGHTER: 1,
            UnitTypeId.BANSHEE: 1,
            UnitTypeId.RAVEN: 0,
            UnitTypeId.BATTLECRUISER: 1,
        }
        self.build_order = list()
        self.evoked = dict()
        self.recon_manager.reset()
        self.a=0
        print("왼쪽 = ",self.start_location)
        print("오른쪽 = ",self.enemy_start_locations)

    async def on_step(self, iteration: int):       
        # print(self.state.score.total_damage_dealt_life)
        
        actions = list()
        
        cc = self.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.get_available_abilities(cc)
        ravens = self.units(UnitTypeId.RAVEN)
        ghosts = self.units(UnitTypeId.GHOST)
        
        if ravens.amount == 0:
            if self.can_afford(UnitTypeId.RAVEN) and self.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
                # 고스트가 하나도 없으면 고스트 훈련
                actions.append(cc.train(UnitTypeId.RAVEN))

        elif ghosts.amount == 0:
            actions += await self.nuke_manager.step()

            

        actions += await self.recon_manager.step()

        await self.do_actions(actions)
        """
        actions = list() # 이번 step에 실행할 액션 목록

        ccs = self.units(UnitTypeId.COMMANDCENTER).idle  # 전체 유닛에서 사령부 검색
        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치

        # 부족한 유닛 숫자 계산
        unit_counts = dict()
        for unit in self.units:
            unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
        
        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        #
        # 사령부 명령 생성
        #
        if ccs.exists:  # 사령부가 하나이상 존재할 경우
            cc = ccs.first  # 첫번째 사령부 선택
            next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
            if self.can_afford(next_unit) and self.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                print(next_unit)
                print(combat_units.amount)
                actions.append(cc.train(next_unit))
                self.evoked[(cc.tag, 'train')] = self.time


        #
        # 해병 명령 생성
        #

        
          # 해병 검색
        '''for marine in marines:
            enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(marine)  # 가장 가까운 적 유닛'''
        
        attacking = False
        for unit in combat_units:
            if unit.is_attacking:
                attacking = True

        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            if unit.type_id is not UnitTypeId.MEDIVAC:
                if combat_units.amount > 20 :
                    # 전투가능한 유닛 수가 20을 넘으면 적 본진으로 공격
                    actions.append(unit.attack(target))
                    use_stimpack = True
                else:
                    # 적 사령부 방향에 유닛 집결
                    target = self.start_location + 0.25 * (enemy_cc.position - self.start_location)
                    actions.append(unit.attack(target))
                    use_stimpack = False

                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if use_stimpack and unit.distance_to(target) < 15:
                        # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

            if unit.type_id is UnitTypeId.MEDIVAC:
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    actions.append(unit.move(combat_units.center))

        #for tank in self.unit
        """


        await self.do_actions(actions)


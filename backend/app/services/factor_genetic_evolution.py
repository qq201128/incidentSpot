"""
因子遗传算法进化器 - 自动生成新因子
"""
from __future__ import annotations

import random
import re
from typing import Any

import numpy as np


class FactorGeneticAlgorithm:
    """
    遗传算法因子进化器

    流程：
    1. 初始种群：现有高表现因子
    2. 选择：基于适应度选择父代
    3. 交叉：组合两个因子的特征
    4. 变异：随机修改参数
    5. 评估：回测并选择优秀后代
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
    ):
        """
        Args:
            population_size: 种群大小
            elite_ratio: 精英比例（直接保留）
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def evolve(
        self,
        parent_factors: list[dict[str, Any]],
        generations: int = 10,
        fitness_func: callable = None,
    ) -> list[dict[str, Any]]:
        """
        进化因子

        Args:
            parent_factors: 父代因子
            generations: 进化代数
            fitness_func: 适应度函数（如果为None，使用默认）

        Returns:
            进化后的因子列表
        """
        if not fitness_func:
            fitness_func = self._default_fitness

        # 初始化种群
        population = self._initialize_population(parent_factors)

        best_factor = None
        best_fitness = -float('inf')

        for gen in range(generations):
            # 1. 评估适应度
            fitness_scores = [fitness_func(f) for f in population]

            # 2. 记录最优
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_factor = population[gen_best_idx].copy()

            # 3. 选择（精英保留 + 轮盘赌）
            elite_count = int(len(population) * self.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elites = [population[i] for i in elite_indices]

            # 4. 生成新一代
            offspring = []
            while len(offspring) < self.population_size - elite_count:
                # 选择父代
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # 变异
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                offspring.extend([child1, child2])

            # 5. 组成新种群
            population = elites + offspring[:self.population_size - elite_count]

        return population

    def _initialize_population(self, parent_factors: list[dict]) -> list[dict]:
        """初始化种群"""
        population = []

        # 添加父代
        population.extend([f.copy() for f in parent_factors[:self.population_size]])

        # 如果不够，生成变异版本
        while len(population) < self.population_size:
            base = random.choice(parent_factors)
            mutated = self._mutate(base.copy())
            population.append(mutated)

        return population

    def _tournament_selection(
        self,
        population: list[dict],
        fitness_scores: list[float],
        tournament_size: int = 3
    ) -> dict:
        """锦标赛选择"""
        indices = random.sample(range(len(population)), tournament_size)
        winner_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[winner_idx].copy()

    def _crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        """
        交叉：组合两个因子的特征

        策略：
        - 公式交叉：交换公式的部分
        - 参数交叉：交换参数
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 公式交叉
        formula1 = parent1.get("formula", "")
        formula2 = parent2.get("formula", "")

        if formula1 and formula2:
            child1_formula, child2_formula = self._crossover_formulas(formula1, formula2)
            child1["formula"] = child1_formula
            child2["formula"] = child2_formula

        # 生成新名称
        child1["name"] = f"evolved_{random.randint(1000, 9999)}"
        child2["name"] = f"evolved_{random.randint(1000, 9999)}"

        return child1, child2

    def _crossover_formulas(self, formula1: str, formula2: str) -> tuple[str, str]:
        """交叉两个公式"""
        # 按运算符分割
        parts1 = self._split_formula(formula1)
        parts2 = self._split_formula(formula2)

        if not parts1 or not parts2:
            return formula1, formula2

        # 单点交叉
        min_len = min(len(parts1), len(parts2))
        if min_len > 1:
            crossover_point = random.randint(1, min_len - 1)

            child1_parts = parts1[:crossover_point] + parts2[crossover_point:]
            child2_parts = parts2[:crossover_point] + parts1[crossover_point:]

            child1_formula = " ".join(child1_parts)
            child2_formula = " ".join(child2_parts)

            return child1_formula, child2_formula

        return formula1, formula2

    def _split_formula(self, formula: str) -> list[str]:
        """分割公式为可交换的部分"""
        # 简单分割（按加减号）
        parts = re.split(r'([+\-])', formula)
        return [p.strip() for p in parts if p.strip()]

    def _mutate(self, factor: dict) -> dict:
        """
        变异：随机修改因子

        策略：
        - 参数变异：调整数值参数
        - 运算符变异：改变运算符
        - 指标变异：替换指标
        """
        mutated = factor.copy()
        formula = mutated.get("formula", "")

        if not formula:
            return mutated

        mutation_type = random.choice(["parameter", "operator", "indicator"])

        if mutation_type == "parameter":
            mutated["formula"] = self._mutate_parameters(formula)
        elif mutation_type == "operator":
            mutated["formula"] = self._mutate_operators(formula)
        else:
            mutated["formula"] = self._mutate_indicators(formula)

        # 生成新名称
        mutated["name"] = f"mutated_{random.randint(1000, 9999)}"

        return mutated

    def _mutate_parameters(self, formula: str) -> str:
        """变异参数（调整数值）"""
        # 查找所有数字
        numbers = re.findall(r'\d+', formula)

        if not numbers:
            return formula

        # 随机选择一个数字进行变异
        old_num = random.choice(numbers)
        old_val = int(old_num)

        # 变异幅度：±20%
        delta = int(old_val * random.uniform(-0.2, 0.2))
        new_val = max(1, old_val + delta)

        # 替换（只替换第一次出现）
        new_formula = formula.replace(old_num, str(new_val), 1)

        return new_formula

    def _mutate_operators(self, formula: str) -> str:
        """变异运算符"""
        operators = ['+', '-', '*', '/']

        # 查找公式中的运算符
        for old_op in operators:
            if old_op in formula:
                # 随机选择新运算符
                new_op = random.choice([op for op in operators if op != old_op])
                # 替换第一个
                new_formula = formula.replace(old_op, new_op, 1)
                return new_formula

        return formula

    def _mutate_indicators(self, formula: str) -> str:
        """变异指标（替换技术指标）"""
        indicator_replacements = {
            'ema': 'sma',
            'sma': 'ema',
            'rsi': 'stoch',
            'std': 'atr',
            'atr': 'std',
        }

        formula_lower = formula.lower()

        for old_ind, new_ind in indicator_replacements.items():
            if old_ind in formula_lower:
                # 保留大小写
                pattern = re.compile(re.escape(old_ind), re.IGNORECASE)
                new_formula = pattern.sub(new_ind, formula, count=1)
                return new_formula

        return formula

    def _default_fitness(self, factor: dict) -> float:
        """
        默认适应度函数

        综合考虑：
        - 胜率
        - IR
        - 夏普比率
        - 交易次数
        """
        win_rate = factor.get("win_rate", 0.5)
        ir = factor.get("ir", 0)
        sharpe = factor.get("sharpe", 0)
        trades = factor.get("trades", 0)

        # 归一化交易次数得分（理想范围：20-100次）
        trades_score = min(1.0, max(0, (trades - 10) / 90))

        # 综合适应度
        fitness = (
            0.4 * win_rate +
            0.3 * min(1.0, ir / 0.5) +
            0.2 * min(1.0, sharpe / 2.0) +
            0.1 * trades_score
        )

        return fitness


def evolve_factors_from_best(
    factors: list[dict[str, Any]],
    top_k: int = 10,
    generations: int = 10,
) -> list[dict[str, Any]]:
    """
    从最优因子进化新因子

    Args:
        factors: 因子库
        top_k: 选择前k个因子作为父代
        generations: 进化代数

    Returns:
        进化后的因子列表
    """
    # 选择表现最好的因子
    sorted_factors = sorted(
        factors,
        key=lambda f: f.get("ir", 0) * 0.6 + f.get("win_rate", 0) * 0.4,
        reverse=True
    )
    parent_factors = sorted_factors[:top_k]

    # 进化
    ga = FactorGeneticAlgorithm(population_size=20)
    evolved = ga.evolve(parent_factors, generations=generations)

    return evolved

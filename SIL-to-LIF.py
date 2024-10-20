import numpy as np
import matplotlib.pyplot as plt

# LIF Neuron 클래스 정의
class LifNeuron:
    def __init__(self, mv_reset=-70, mv_threshold=-55, mv_rest=-65, tau=0.1, g=10):
        """
        LIF 뉴런의 초기 설정을 담당합니다.

        Parameters:
        - mv_reset: 스파이크 후 리셋 전압 (mV)
        - mv_threshold: 스파이크 임계 전압 (mV)
        - mv_rest: 휴지 상태 전압 (mV)
        - tau: 시간 상수 (초) - leakage를 줄이기 위해 10배 증가 (기본값: 0.1 s)
        - g: 전도도 (nS)
        """
        self.mv_reset = mv_reset        # Reset 전압 (mV)
        self.mv_threshold = mv_threshold  # 스파이크 임계 전압 (mV)
        self.mv_rest = mv_rest          # 휴지 상태 전압 (mV)
        self.tau = tau                  # 시간 상수 (초)
        self.g = g                      # 전도도 (nS)

        self.membrane_potential = self.mv_rest  # 현재 막 전위 초기화

    def step(self, input_current_pa=0, dt=0.001):
        """
        LIF 뉴런의 한 스텝을 시뮬레이션합니다.

        Parameters:
        - input_current_pa: 입력 전류 (pA)
        - dt: 시간 간격 (초)

        Returns:
        - infos: dict containing 'membrane_potential' and 'is_spike'
        """
        is_spike = False
        dV = (-(self.membrane_potential - self.mv_rest) / self.tau + self.g * input_current_pa) * dt
        self.membrane_potential += dV

        if self.membrane_potential >= self.mv_threshold:
            is_spike = True
            self.membrane_potential = self.mv_reset

        infos = {
            "membrane_potential": self.membrane_potential,
            "is_spike": is_spike
        }

        return infos

# 시뮬레이션 파라미터 설정
frequency = 50  # Hz, 오실레이터의 기본 주파수
injection_strength = 0.5  # 주입된 파형의 강도
phase_shift = np.pi / 6  # 오실레이터와 주입된 파형 사이의 위상 차이
amplitude = 1  # 오실레이터의 진폭
duration = 2  # 초
sampling_rate = 1000  # 초당 샘플 수

# 유도된 파라미터
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)  # 시간 벡터

# 도플러 효과를 위한 변수 (움직이는 물체 시나리오)
object_velocity = 10  # m/s (예: 물체의 속도)
c = 340  # m/s (음속, 도플러 효과 계산을 위한 속도)

# 도플러 효과 계산 함수
def doppler_shift(frequency, velocity, c):
    """
    도플러 효과에 의한 주파수 변화를 계산합니다.
    물체가 레이더를 향해 접근할 때의 주파수.
    """
    return frequency * (c / (c - velocity))

# 시나리오 정의
scenarios = {
    0: "No Object",
    1: "Static Object",
    2: "Moving Object with Doppler Shift"
}

# 각 시나리오에 대한 주입 파형 생성 함수
def generate_injected_wave(scenario, t, frequency, injection_strength, amplitude, phase_shift, object_velocity, c):
    if scenario == 0:
        # 물체 없음: 주입 신호 없음
        return np.zeros_like(t)
    elif scenario == 1:
        # 물체 정지: 주입 신호가 고정된 주파수와 위상
        return injection_strength * amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    elif scenario == 2:
        # 물체 이동: 도플러 효과에 따른 주파수 변화
        doppler_freq = doppler_shift(frequency, object_velocity, c)
        return injection_strength * amplitude * np.sin(2 * np.pi * doppler_freq * t + phase_shift)
    else:
        raise ValueError("Invalid scenario selected.")

# 오실레이터 파형 생성
oscillator_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# PNG 파일로 저장할 그림과 서브플롯 생성
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('LIF Neuron Response to SIL Radar Signals (Amplified by 5x, Leak Reduced by 10x)', fontsize=20)

# 각 시나리오별로 서브플롯 생성
for scenario, name in scenarios.items():
    # 주입 신호 생성
    injection_wave = generate_injected_wave(scenario, t, frequency, injection_strength, amplitude, phase_shift, object_velocity, c)
    # 결합된 파형
    combined_wave = oscillator_wave + injection_wave

    # 신호 증폭 (5배)
    combined_wave_amplified = combined_wave * 23

    # LIF 뉴런 초기화 (tau=0.1으로 설정하여 leakage 감소)
    neuron = LifNeuron(tau=0.1)  # tau를 10배 증가시켜 leakage 감소 (0.1 s)

    # 입력 전류로 변환 (스케일링)
    # 주입 신호의 진폭을 전류 범위로 스케일링 (예: 10 pA per unit)
    input_current_pa = combined_wave_amplified * 10  # 스케일링 팩터 조정 가능

    # 시뮬레이션 결과 저장 리스트
    membrane_potentials = []
    spikes = []

    # 시뮬레이션 루프
    for i in range(len(t)):
        info = neuron.step(input_current_pa=input_current_pa[i], dt=1/sampling_rate)
        membrane_potentials.append(info["membrane_potential"])
        spikes.append(info["is_spike"])

    # 스파이크 이벤트 시간 추출
    spike_times = t[np.array(spikes)]

    # 첫 번째 서브플롯: 입력 전류
    ax_input = axes[scenario, 0]
    ax_input.plot(t, input_current_pa, label='Input Current (pA)', color='blue')
    ax_input.set_xlim(0, 1)  # 1초 범위 표시
    ax_input.set_ylim(np.min(input_current_pa)-1, np.max(input_current_pa)+1)
    ax_input.set_xlabel('Time [s]')
    ax_input.set_ylabel('Input Current (pA)')
    ax_input.set_title(f'Scenario: {name} - Input Current (Amplified by 5x)')
    ax_input.legend(loc='upper right')
    ax_input.grid(True)

    # 두 번째 서브플롯: LIF 뉴런의 막 전위와 스파이크
    ax_potential = axes[scenario, 1]
    ax_potential.plot(t, membrane_potentials, label='Membrane Potential (mV)', color='green')
    ax_potential.scatter(spike_times, [neuron.mv_threshold]*len(spike_times), color='red', marker='o', label='Spikes')
    ax_potential.set_xlim(0, 1)  # 1초 범위 표시
    ax_input.set_ylim(-400, 400)  # y축 상한을 350, 하한을 -350으로 고정
    ax_potential.set_ylim(neuron.mv_reset - 5, neuron.mv_threshold + 5)
    ax_potential.set_xlabel('Time [s]')
    ax_potential.set_ylabel('Membrane Potential (mV)')
    ax_potential.set_title(f'Scenario: {name} - LIF Neuron Response')
    ax_potential.legend(loc='upper right')
    ax_potential.grid(True)

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# PNG 파일로 저장
plt.savefig('lif_neuron_response_scenarios_amplified_leak_reduced.png')
plt.show()

print("PNG 파일 'lif_neuron_response_scenarios_amplified_leak_reduced.png'가 생성되었습니다.")

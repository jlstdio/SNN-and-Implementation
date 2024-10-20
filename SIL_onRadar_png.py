import numpy as np
import matplotlib.pyplot as plt

# 시뮬레이션 파라미터
frequency = 50  # Hz, 오실레이터의 기본 주파수
injection_strength = 0.5  # 주입된 파형의 강도
phase_shift = np.pi / 6  # 오실레이터와 주입된 파형 사이의 위상 차이
amplitude = 1  # 오실레이터의 진폭
duration = 2  # 초
sampling_rate = 1000  # 초당 샘플 수

# 유도된 파라미터
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

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


# 각 시나리오에 대한 주입 파형 생성
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


# 오실레이터 파형
oscillator_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# 그림과 축 생성
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle('Self-Injection Locking Simulation in Radar Systems', fontsize=16)

# 각 시나리오별로 서브플롯 생성
for scenario, name in scenarios.items():
    ax = axes[scenario]
    injection_wave = generate_injected_wave(scenario, t, frequency, injection_strength, amplitude, phase_shift,
                                            object_velocity, c)
    combined_wave = oscillator_wave + injection_wave

    ax.plot(t, oscillator_wave, label='Oscillator Wave', color='blue')
    ax.plot(t, injection_wave, label='Injected Wave', color='orange')
    ax.plot(t, combined_wave, label='Combined Wave', color='green')
    ax.axvline(x=0, color='red', linestyle='--', label='Current Time')

    ax.set_xlim(0, 1)  # 1초 범위 표시
    ax.set_ylim(-2 * amplitude, 2 * amplitude)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Scenario: {name}')
    ax.legend(loc='upper right')
    ax.grid(True)

# 레이아웃 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# PNG 파일로 저장
plt.savefig('self_injection_locking_scenarios.png')
plt.show()

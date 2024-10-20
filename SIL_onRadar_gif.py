import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


# 애니메이션을 생성하고 저장하는 함수
def create_animation(scenario, scenario_name, oscillator_wave, t, injection_wave, combined_wave):
    # 그림과 축 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)  # 파형의 1초를 표시
    ax.set_ylim(-2 * amplitude, 2 * amplitude)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Self-Injection Locking Simulation - {scenario_name}')

    # 오실레이터, 주입된, 결합된 파형을 위한 초기 라인 설정
    line_osc, = ax.plot([], [], label='Oscillator Wave', color='blue')
    line_inj, = ax.plot([], [], label='Injected Wave', color='orange')
    line_combined, = ax.plot([], [], label='Combined Wave', color='green')

    # 현재 시간을 나타내는 수직선
    current_time_line = ax.axvline(x=0, color='red', linestyle='--', label='Current Time')

    # 범례 설정
    ax.legend(loc='upper right')

    # 프레임 수와 윈도우 크기 설정
    frames = len(t)
    window_size = int(sampling_rate * 0.02)  # 20 ms 윈도우

    def init():
        """애니메이션의 초기 상태를 설정합니다."""
        line_osc.set_data([], [])
        line_inj.set_data([], [])
        line_combined.set_data([], [])
        current_time_line.set_xdata([0, 0])  # 시퀀스로 변경
        return [line_osc, line_inj, line_combined, current_time_line]

    def animate(i):
        """애니메이션의 각 프레임을 업데이트합니다."""
        # 표시할 데이터의 윈도우 정의
        if i < window_size:
            start = 0
            end = window_size
        else:
            start = i - window_size
            end = i

        # 현재 프레임이 끝을 초과하지 않도록 조정
        if end > len(t):
            end = len(t)
            start = end - window_size if end - window_size >= 0 else 0

        # 오실레이터 파형 업데이트
        line_osc.set_data(t[start:end], oscillator_wave[start:end])

        # 주입된 파형 업데이트
        line_inj.set_data(t[start:end], injection_wave[start:end])

        # 결합된 파형 업데이트
        line_combined.set_data(t[start:end], combined_wave[start:end])

        # 현재 시간 선 업데이트 (시퀀스로 변경)
        current_x = t[i] if i < len(t) else t[-1]
        current_time_line.set_xdata([current_x, current_x])

        return [line_osc, line_inj, line_combined, current_time_line]

    # 애니메이션 생성
    ani = animation.FuncAnimation(
        fig, animate, frames=frames,
        init_func=init, blit=False, interval=20, repeat=False
    )

    # GIF로 저장
    gif_filename = f'self_injection_locking_{scenario_name.replace(" ", "_").lower()}.gif'
    ani.save(gif_filename, writer='pillow', fps=50)
    print(f'Animation saved as {gif_filename}')

    plt.close(fig)  # 창을 닫아줍니다 (필요 시)


# 시나리오별로 애니메이션 생성
for scenario, name in scenarios.items():
    injection_wave = generate_injected_wave(scenario, t, frequency, injection_strength, amplitude, phase_shift,
                                            object_velocity, c)
    combined_wave = oscillator_wave + injection_wave
    create_animation(scenario, name, oscillator_wave, t, injection_wave, combined_wave)

print("All animations have been created and saved as GIF files.")

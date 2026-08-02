from sft_12hz_lora_mlflow_multi_speaker import (
    build_first_ref_audio_map,
    build_speaker_slot_map,
)


def main():
    train_data = [
        {"speaker_id": "spk_115", "ref_audio": "./audio/b.wav"},
        {"speaker_id": "spk_021", "ref_audio": "./audio/a.wav"},
        {"speaker_id": "spk_115", "ref_audio": "./audio/b.wav"},
    ]

    assert build_speaker_slot_map(train_data, start_slot=3000) == {
        "spk_021": 3000,
        "spk_115": 3001,
    }

    assert build_first_ref_audio_map(train_data) == {
        "spk_115": "./audio/b.wav",
        "spk_021": "./audio/a.wav",
    }

    try:
        build_first_ref_audio_map(
            [
                {"speaker_id": "spk_001", "ref_audio": "./audio/1.wav"},
                {"speaker_id": "spk_001", "ref_audio": "./audio/2.wav"},
            ]
        )
    except ValueError as exc:
        assert "conflicting ref_audio" in str(exc)
    else:
        raise AssertionError("expected ValueError for conflicting ref_audio values")


if __name__ == "__main__":
    main()

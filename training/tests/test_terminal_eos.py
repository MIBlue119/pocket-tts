"""Every utterance needs an EOS target after, not instead of, its last audio frame."""

import pytest
import torch

from pocket_tts.models.flow_lm import FlowNet
from training.tests.test_smoke import LDIM, tiny_model


@pytest.mark.parametrize("lengths", [(3,), (3, 1)])
def test_terminal_eos_after_each_last_audio_frame(
    lengths: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tiny_model("lsd").eval()
    model.args.text_dropout = model.args.voice_dropout = 0.0
    latents = torch.randn(len(lengths), max(lengths), LDIM)
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    original_mask = mask.clone()
    tokens = [torch.tensor([1, 2])] * len(lengths)
    voice = torch.empty(len(lengths), 0, LDIM)
    logits: list[torch.Tensor] = []
    audio_inputs: list[torch.Tensor] = []
    flow_targets: list[torch.Tensor] = []

    def capture_eos(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        output.retain_grad()
        logits.append(output)

    model.flow_lm.out_eos.register_forward_hook(capture_eos)
    model.flow_lm.input_linear.register_forward_pre_hook(
        lambda module, inputs: audio_inputs.append(inputs[0].detach().clone())
    )
    original_loss = model.flow.loss

    def capture_flow(
        head: FlowNet, noise: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        flow_targets.append(target.detach().clone())
        return original_loss(head, noise, target)

    monkeypatch.setattr(model.flow, "loss", capture_flow)
    loss, metrics = model(
        latents, mask, tokens, voice, num_voice_prompt_frames=torch.zeros(len(lengths))
    )
    # Use the actual forward graph, rather than reconstructing EOS labels in the test.
    loss.backward()
    assert logits[-1].grad is not None
    gradient = logits[-1].grad[..., 0]
    assert gradient.shape == (len(lengths), max(lengths) + 1)
    for row, length in enumerate(lengths):
        assert torch.all(gradient[row, :length] > 0)  # Real audio predicts continuation.
        assert gradient[row, length] < 0  # Gradient descent increases terminal EOS.
        assert torch.all(gradient[row, length + 1 :] == 0)
    normalized = (latents - model.flow_lm.emb_mean) / model.flow_lm.emb_std
    assert torch.equal(audio_inputs[-1][:, 1:], normalized)
    expected_targets = normalized[original_mask].repeat(model.args.flow_batch_multiplier, 1)
    torch.testing.assert_close(flow_targets[-1], expected_targets)
    assert torch.equal(mask, original_mask)
    assert torch.isfinite(metrics["eos_loss"])

---
description: 检查服务器评测进度和结果
---

## 检查 MedAgent 评测状态

服务器信息：
- SSH: `sshpass -p 'YfexT9sxuGWf' ssh -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -p 50556 root@connect.westb.seetacloud.com`
- 项目路径: `/root/autodl-tmp/med-agent`
- 日志: `/root/autodl-tmp/eval_all.log`

### 步骤

// turbo
1. 查看评测日志最后 30 行，确认当前进度：
```bash
sshpass -p 'YfexT9sxuGWf' ssh -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -p 50556 root@connect.westb.seetacloud.com 'tail -30 /root/autodl-tmp/eval_all.log 2>/dev/null'
```

// turbo
2. 查看已完成的评测报告：
```bash
sshpass -p 'YfexT9sxuGWf' ssh -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -p 50556 root@connect.westb.seetacloud.com 'for m in base sft rest; do echo "=== $m ==="; cat /root/autodl-tmp/med-agent/results/$m/evaluation_report.json 2>/dev/null | python3 -m json.tool || echo "(未完成)"; done'
```

3. 根据输出汇总进度并告知用户：
   - 哪些模型已评测完成
   - 当前正在评测哪个模型、进度百分比
   - 预计剩余时间
   - 如有错误，分析原因并给出修复建议

# Getting Your Tinker API Key

Tinker is a hosted platform for fine-tuning language models. You need a Tinker account and API key to use this training setup.

## Sign Up Process

### 1. Visit Tinker Website

Go to: **https://tinker-docs.thinkingmachines.ai/**

### 2. Create Account

1. Click "Sign Up" or "Get Started"
2. Fill in your details:
   - Email address
   - Password
   - Organization name (optional)
3. Verify your email address

### 3. Get Your API Key

After signing up and logging in:

1. Navigate to your **Dashboard**
2. Go to **Settings** or **API Keys** section
3. Click **"Create New API Key"** or **"Generate API Key"**
4. Copy your API key (starts with `tinker_...`)
5. **Important:** Save it securely - you won't be able to see it again!

### 4. Add to .env

```bash
cd finetuning/tinker
echo "TINKER_API_KEY=your_actual_api_key_here" >> .env
```

Or run the interactive setup:
```bash
./setup_api_keys.sh
```

## Pricing & Plans

Tinker offers different pricing tiers:

### Free Tier (if available)
- Limited GPU hours
- Good for testing and small experiments
- May have queue times

### Paid Plans
- More GPU hours
- Priority queue access
- Larger model support
- Better support

**Check current pricing:** https://tinker-docs.thinkingmachines.ai/pricing

## Alternative: Self-Hosted Training

If you don't want to use Tinker's hosted service, you can:

### Option 1: Use Original Implementation
Use the non-Tinker training script:
```bash
cd ../scripts
python train_lora.py \
  --model-name Qwen/Qwen2.5-3B \
  --data-dir ../data \
  --output-dir ../models/qwen-lora
```

This runs locally on your machine without requiring Tinker API.

### Option 2: Use HuggingFace Trainer Directly
The original `train_lora.py` uses HuggingFace's Trainer directly:
- No API key required
- Runs on your local GPU or CPU
- Full control over training

See: `finetuning/README.md` for the original setup.

## Tinker vs Local Training

| Aspect | Tinker (Hosted) | Local Training |
|--------|----------------|----------------|
| **Setup** | API key only | Full GPU setup |
| **Hardware** | Cloud GPUs | Your hardware |
| **Cost** | Pay per use | One-time GPU cost |
| **Scalability** | Easy scaling | Limited by hardware |
| **Control** | Platform managed | Full control |
| **Internet** | Required | Optional |

## Support

### Tinker Support
- Documentation: https://tinker-docs.thinkingmachines.ai/
- Support email: Check their website
- Community: Discord/Slack (if available)

### GPSBench Support
- For GPSBench-specific issues: GitHub Issues
- For training setup: See `README.md` and `API_KEYS_GUIDE.md`

## FAQ

### Q: Is Tinker free?
Check their current pricing at: https://tinker-docs.thinkingmachines.ai/pricing

They may offer:
- Free tier for testing
- Academic discounts
- Trial credits for new users

### Q: Can I train locally instead?
Yes! Use the original implementation:
```bash
cd ../scripts
python train_lora.py --help
```

### Q: Where do I find my API key?
After signing up:
1. Log into Tinker dashboard
2. Go to Settings → API Keys
3. Copy your key

### Q: My API key isn't working
1. Check it's copied correctly (no extra spaces)
2. Verify your account is activated
3. Check if your free credits ran out
4. Contact Tinker support

### Q: Can I use multiple API keys?
Yes, for different environments:
```bash
# Development
TINKER_API_KEY=tinker_dev_key

# Production
TINKER_API_KEY=tinker_prod_key
```

### Q: Is my API key secure?
Yes, as long as:
- You keep `.env` file private
- Don't commit it to git (it's in `.gitignore`)
- Don't share your key publicly
- Rotate keys regularly

## Next Steps

After getting your API key:

1. ✅ Add to `.env` file
2. ✅ Test with: `python train.py max_samples=10 max_steps=5`
3. ✅ Run full training: `./quickstart.sh`

Need help? See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md)

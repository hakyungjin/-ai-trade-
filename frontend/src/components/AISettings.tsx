import { useState, useEffect } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { aiApi, settingsApi } from '../api/client';
import { Send, Save, RotateCcw, Sparkles, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';

export function AISettings() {
  const { settings, setSettings } = useTradingStore();

  const [prompt, setPrompt] = useState('');
  const [parsedRule, setParsedRule] = useState<{
    stop_loss?: number;
    take_profit?: number;
    max_position?: number;
    trailing_stop?: number;
    description: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const res = await settingsApi.get();
        setSettings(res.data);
      } catch (error) {
        console.error('설정 로드 실패:', error);
      }
    };
    loadSettings();
  }, [setSettings]);

  const handlePromptSubmit = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    try {
      const res = await aiApi.parsePrompt(prompt);
      setParsedRule(res.data);

      if (res.data.stop_loss || res.data.take_profit || res.data.max_position) {
        const updates: Record<string, unknown> = {};
        if (res.data.stop_loss) {
          updates.default_stop_loss = res.data.stop_loss;
        }
        if (res.data.take_profit) {
          updates.default_take_profit = res.data.take_profit;
        }
        if (res.data.max_position) {
          updates.max_position_size = res.data.max_position;
        }
        if (res.data.trailing_stop) {
          updates.trailing_stop_percent = res.data.trailing_stop;
          updates.trailing_stop_enabled = true;
        }
        setSettings({ ...settings, ...updates } as typeof settings);
      }
    } catch (error) {
      console.error('프롬프트 파싱 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async () => {
    try {
      await settingsApi.update(settings as any);
      setSaveMessage('설정이 저장되었습니다');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (error) {
      console.error('설정 저장 실패:', error);
      setSaveMessage('저장 실패');
    }
  };

  const handleResetSettings = async () => {
    try {
      const res = await settingsApi.reset();
      setSettings(res.data.settings);
      setSaveMessage('설정이 초기화되었습니다');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (error) {
      console.error('설정 초기화 실패:', error);
    }
  };

  const handleSettingChange = (key: string, value: unknown) => {
    setSettings({ ...settings, [key]: value });
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      <h1 className="text-2xl font-bold">AI 설정</h1>

      {/* Prompt Input */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            자연어 설정
          </CardTitle>
          <CardDescription>
            자연어로 거래 규칙을 설정하세요. 예: "스탑로스 3%, 익절 5%로 설정해줘"
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handlePromptSubmit()}
              placeholder="예: 스탑로스 3%, 익절 5%로 설정해줘"
              className="h-12"
            />
            <Button
              onClick={handlePromptSubmit}
              disabled={loading}
              size="icon"
              className="h-12 w-12"
            >
              <Send className="w-5 h-5" />
            </Button>
          </div>

          {parsedRule && (
            <Alert className="bg-primary/10 border-primary/30">
              <Sparkles className="h-4 w-4" />
              <AlertDescription>
                <span className="font-medium">파싱 결과: </span>
                {parsedRule.description}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Manual Settings */}
      <Card>
        <CardHeader>
          <CardTitle>거래 설정</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Stop Loss & Take Profit */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>기본 스탑로스</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={(settings.default_stop_loss * 100).toFixed(1)}
                  onChange={(e) =>
                    handleSettingChange('default_stop_loss', parseFloat(e.target.value) / 100)
                  }
                  className="h-11 text-right"
                />
                <span className="text-muted-foreground w-8">%</span>
              </div>
            </div>
            <div className="space-y-2">
              <Label>기본 익절</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={(settings.default_take_profit * 100).toFixed(1)}
                  onChange={(e) =>
                    handleSettingChange('default_take_profit', parseFloat(e.target.value) / 100)
                  }
                  className="h-11 text-right"
                />
                <span className="text-muted-foreground w-8">%</span>
              </div>
            </div>
          </div>

          <Separator />

          {/* Position & Threshold */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>최대 포지션 크기</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={settings.max_position_size}
                  onChange={(e) =>
                    handleSettingChange('max_position_size', parseFloat(e.target.value))
                  }
                  className="h-11 text-right"
                />
                <span className="text-muted-foreground w-12">USDT</span>
              </div>
            </div>
            <div className="space-y-2">
              <Label>AI 신뢰도 임계값</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={(settings.prediction_threshold * 100).toFixed(0)}
                  onChange={(e) =>
                    handleSettingChange('prediction_threshold', parseFloat(e.target.value) / 100)
                  }
                  className="h-11 text-right"
                />
                <span className="text-muted-foreground w-8">%</span>
              </div>
            </div>
          </div>

          <Separator />

          {/* Toggles */}
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/50">
              <div className="space-y-0.5">
                <Label htmlFor="autoTrading" className="cursor-pointer">자동 매매</Label>
                <p className="text-sm text-muted-foreground">AI 신호에 따라 자동으로 거래</p>
              </div>
              <Switch
                id="autoTrading"
                checked={settings.auto_trading_enabled}
                onCheckedChange={(checked) =>
                  handleSettingChange('auto_trading_enabled', checked)
                }
              />
            </div>

            <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/50">
              <div className="space-y-0.5">
                <Label htmlFor="trailingStop" className="cursor-pointer">트레일링 스탑</Label>
                <p className="text-sm text-muted-foreground">수익 보호를 위한 추적 손절</p>
              </div>
              <div className="flex items-center gap-3">
                <Switch
                  id="trailingStop"
                  checked={settings.trailing_stop_enabled}
                  onCheckedChange={(checked) =>
                    handleSettingChange('trailing_stop_enabled', checked)
                  }
                />
                {settings.trailing_stop_enabled && (
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      value={(settings.trailing_stop_percent * 100).toFixed(1)}
                      onChange={(e) =>
                        handleSettingChange(
                          'trailing_stop_percent',
                          parseFloat(e.target.value) / 100
                        )
                      }
                      className="w-20 h-9 text-center"
                    />
                    <span className="text-muted-foreground">%</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <Separator />

          {/* Daily Limits */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>일일 최대 거래 횟수</Label>
              <Input
                type="number"
                value={settings.max_daily_trades}
                onChange={(e) =>
                  handleSettingChange('max_daily_trades', parseInt(e.target.value))
                }
                className="h-11 text-right"
              />
            </div>
            <div className="space-y-2">
              <Label>일일 최대 손실</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={(settings.max_daily_loss * 100).toFixed(0)}
                  onChange={(e) =>
                    handleSettingChange('max_daily_loss', parseFloat(e.target.value) / 100)
                  }
                  className="h-11 text-right"
                />
                <span className="text-muted-foreground w-8">%</span>
              </div>
            </div>
          </div>

          {/* Save Message */}
          {saveMessage && (
            <Alert className="bg-green-500/10 border-green-500/30 text-green-500">
              <CheckCircle2 className="h-4 w-4" />
              <AlertDescription>{saveMessage}</AlertDescription>
            </Alert>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 pt-2">
            <Button onClick={handleSaveSettings} className="flex-1 h-12">
              <Save className="w-4 h-4 mr-2" />
              저장
            </Button>
            <Button onClick={handleResetSettings} variant="outline" className="h-12">
              <RotateCcw className="w-4 h-4 mr-2" />
              초기화
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

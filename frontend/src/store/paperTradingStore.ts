import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type MarketType = 'spot' | 'futures';
export type PositionType = 'LONG' | 'SHORT';  // 선물
export type SpotSide = 'BUY' | 'SELL';  // 현물
export type PositionStatus = 'OPEN' | 'CLOSED';

export interface PaperPosition {
  id: string;
  symbol: string;
  marketType: MarketType;
  type: PositionType;  // 선물: LONG/SHORT
  side?: SpotSide;     // 현물: BUY (보유중인 자산)
  entryPrice: number;
  quantity: number;
  leverage: number;    // 현물은 항상 1
  openedAt: string;
  closedAt?: string;
  closePrice?: number;
  pnl?: number;
  pnlPercent?: number;
  status: PositionStatus;
}

export interface PaperTrade {
  id: string;
  positionId: string;
  symbol: string;
  marketType: MarketType;
  type: PositionType;
  side?: SpotSide;
  action: 'OPEN' | 'CLOSE';
  price: number;
  quantity: number;
  timestamp: string;
  pnl?: number;
}

interface PaperTradingState {
  // 잔고 (가상 USDT)
  balance: number;
  initialBalance: number;
  
  // 열린 포지션
  positions: PaperPosition[];
  
  // 거래 내역
  trades: PaperTrade[];
  
  // 통계
  totalPnl: number;
  winCount: number;
  loseCount: number;
  
  // Actions
  openFuturesPosition: (symbol: string, type: PositionType, price: number, quantity: number, leverage: number) => boolean;
  closeFuturesPosition: (positionId: string, currentPrice: number) => boolean;
  buySpot: (symbol: string, price: number, quantity: number) => boolean;
  sellSpot: (symbol: string, price: number, quantity: number) => boolean;
  closePosition: (positionId: string, currentPrice: number) => boolean;
  resetAccount: () => void;
}

const generateId = () => Math.random().toString(36).substring(2, 15);

export const usePaperTradingStore = create<PaperTradingState>()(
  persist(
    (set, get) => ({
      balance: 10000, // 시작 잔고 10,000 USDT
      initialBalance: 10000,
      positions: [] as PaperPosition[],
      trades: [] as PaperTrade[],
      totalPnl: 0,
      winCount: 0,
      loseCount: 0,

      // 선물 포지션 오픈 (LONG/SHORT)
      openFuturesPosition: (symbol, type, price, quantity, leverage) => {
        const { balance, positions, trades } = get();
        const cost = price * quantity;
        
        // 필요 마진 계산 (레버리지 적용)
        const requiredMargin = cost / leverage;
        if (requiredMargin > balance) {
          return false;
        }

        const positionId = generateId();
        const newPosition: PaperPosition = {
          id: positionId,
          symbol,
          marketType: 'futures',
          type,
          entryPrice: price,
          quantity,
          leverage,
          openedAt: new Date().toISOString(),
          status: 'OPEN',
        };

        const newTrade: PaperTrade = {
          id: generateId(),
          positionId,
          symbol,
          marketType: 'futures',
          type,
          action: 'OPEN',
          price,
          quantity,
          timestamp: new Date().toISOString(),
        };

        set({
          balance: balance - requiredMargin,
          positions: [...positions, newPosition],
          trades: [...trades, newTrade],
        });

        return true;
      },

      // 선물 포지션 종료
      closeFuturesPosition: (positionId, currentPrice) => {
        const { positions, trades, balance, totalPnl, winCount, loseCount } = get();
        const position = positions.find((p) => p.id === positionId && p.status === 'OPEN' && p.marketType === 'futures');
        
        if (!position) {
          return false;
        }

        // PnL 계산
        let pnl: number;
        if (position.type === 'LONG') {
          pnl = (currentPrice - position.entryPrice) * position.quantity * position.leverage;
        } else {
          pnl = (position.entryPrice - currentPrice) * position.quantity * position.leverage;
        }
        const pnlPercent = (pnl / (position.entryPrice * position.quantity)) * 100;

        // 포지션 종료
        const updatedPositions = positions.map((p) =>
          p.id === positionId
            ? {
                ...p,
                status: 'CLOSED' as PositionStatus,
                closedAt: new Date().toISOString(),
                closePrice: currentPrice,
                pnl,
                pnlPercent,
              }
            : p
        );

        // 거래 기록 추가
        const newTrade: PaperTrade = {
          id: generateId(),
          positionId,
          symbol: position.symbol,
          marketType: 'futures',
          type: position.type,
          action: 'CLOSE',
          price: currentPrice,
          quantity: position.quantity,
          timestamp: new Date().toISOString(),
          pnl,
        };

        // 잔고 복원 + PnL
        const margin = (position.entryPrice * position.quantity) / position.leverage;
        const newBalance = balance + margin + pnl;

        set({
          positions: updatedPositions,
          trades: [...trades, newTrade],
          balance: newBalance,
          totalPnl: totalPnl + pnl,
          winCount: pnl > 0 ? winCount + 1 : winCount,
          loseCount: pnl < 0 ? loseCount + 1 : loseCount,
        });

        return true;
      },

      // 현물 매수
      buySpot: (symbol, price, quantity) => {
        const { balance, positions, trades } = get();
        const cost = price * quantity;
        
        if (cost > balance) {
          return false;
        }

        const positionId = generateId();
        
        // 기존 동일 심볼 포지션 확인 (평균 단가 계산)
        const existingPosition = positions.find(
          (p) => p.symbol === symbol && p.marketType === 'spot' && p.status === 'OPEN'
        );

        if (existingPosition) {
          // 기존 포지션에 추가 (평균 단가 계산)
          const totalQuantity = existingPosition.quantity + quantity;
          const avgPrice = (existingPosition.entryPrice * existingPosition.quantity + price * quantity) / totalQuantity;
          
          const updatedPositions = positions.map((p) =>
            p.id === existingPosition.id
              ? { ...p, quantity: totalQuantity, entryPrice: avgPrice }
              : p
          );

          const newTrade: PaperTrade = {
            id: generateId(),
            positionId: existingPosition.id,
            symbol,
            marketType: 'spot',
            type: 'LONG',
            side: 'BUY',
            action: 'OPEN',
            price,
            quantity,
            timestamp: new Date().toISOString(),
          };

          set({
            balance: balance - cost,
            positions: updatedPositions,
            trades: [...trades, newTrade],
          });
        } else {
          // 새 포지션 생성
          const newPosition: PaperPosition = {
            id: positionId,
            symbol,
            marketType: 'spot',
            type: 'LONG',
            side: 'BUY',
            entryPrice: price,
            quantity,
            leverage: 1,
            openedAt: new Date().toISOString(),
            status: 'OPEN',
          };

          const newTrade: PaperTrade = {
            id: generateId(),
            positionId,
            symbol,
            marketType: 'spot',
            type: 'LONG',
            side: 'BUY',
            action: 'OPEN',
            price,
            quantity,
            timestamp: new Date().toISOString(),
          };

          set({
            balance: balance - cost,
            positions: [...positions, newPosition],
            trades: [...trades, newTrade],
          });
        }

        return true;
      },

      // 현물 매도 (보유 자산 판매)
      sellSpot: (symbol, price, quantity) => {
        const { positions, trades, balance, totalPnl, winCount, loseCount } = get();
        const position = positions.find(
          (p) => p.symbol === symbol && p.marketType === 'spot' && p.status === 'OPEN'
        );

        if (!position || position.quantity < quantity) {
          return false;
        }

        const revenue = price * quantity;
        const cost = position.entryPrice * quantity;
        const pnl = revenue - cost;
        const pnlPercent = (pnl / cost) * 100;

        const remainingQuantity = position.quantity - quantity;

        if (remainingQuantity <= 0) {
          // 전량 매도 - 포지션 종료
          const updatedPositions = positions.map((p) =>
            p.id === position.id
              ? {
                  ...p,
                  status: 'CLOSED' as PositionStatus,
                  closedAt: new Date().toISOString(),
                  closePrice: price,
                  pnl,
                  pnlPercent,
                }
              : p
          );

          const newTrade: PaperTrade = {
            id: generateId(),
            positionId: position.id,
            symbol,
            marketType: 'spot',
            type: 'LONG',
            side: 'SELL',
            action: 'CLOSE',
            price,
            quantity,
            timestamp: new Date().toISOString(),
            pnl,
          };

          set({
            positions: updatedPositions,
            trades: [...trades, newTrade],
            balance: balance + revenue,
            totalPnl: totalPnl + pnl,
            winCount: pnl > 0 ? winCount + 1 : winCount,
            loseCount: pnl < 0 ? loseCount + 1 : loseCount,
          });
        } else {
          // 일부 매도 - 수량만 줄임
          const updatedPositions = positions.map((p) =>
            p.id === position.id
              ? { ...p, quantity: remainingQuantity }
              : p
          );

          const newTrade: PaperTrade = {
            id: generateId(),
            positionId: position.id,
            symbol,
            marketType: 'spot',
            type: 'LONG',
            side: 'SELL',
            action: 'CLOSE',
            price,
            quantity,
            timestamp: new Date().toISOString(),
            pnl,
          };

          set({
            positions: updatedPositions,
            trades: [...trades, newTrade],
            balance: balance + revenue,
            totalPnl: totalPnl + pnl,
            winCount: pnl > 0 ? winCount + 1 : winCount,
            loseCount: pnl < 0 ? loseCount + 1 : loseCount,
          });
        }

        return true;
      },

      // 포지션 종료 (범용)
      closePosition: (positionId, currentPrice) => {
        const { positions } = get();
        const position = positions.find((p) => p.id === positionId && p.status === 'OPEN');
        
        if (!position) return false;

        if (position.marketType === 'futures') {
          return get().closeFuturesPosition(positionId, currentPrice);
        } else {
          return get().sellSpot(position.symbol, currentPrice, position.quantity);
        }
      },

      resetAccount: () => {
        set({
          balance: 10000,
          initialBalance: 10000,
          positions: [],
          trades: [],
          totalPnl: 0,
          winCount: 0,
          loseCount: 0,
        });
      },
    }),
    {
      name: 'paper-trading-storage',
      version: 2, // 버전 업그레이드 - marketType 필드 추가
      migrate: (persistedState: any, version: number) => {
        if (version < 2) {
          // 기존 데이터에 marketType 필드가 없으면 추가
          const state = persistedState as PaperTradingState;
          return {
            ...state,
            positions: (state.positions || []).map((p: any) => ({
              ...p,
              marketType: p.marketType || 'futures', // 기존 데이터는 futures로 간주
            })),
            trades: (state.trades || []).map((t: any) => ({
              ...t,
              marketType: t.marketType || 'futures',
            })),
          };
        }
        return persistedState;
      },
    }
  )
);

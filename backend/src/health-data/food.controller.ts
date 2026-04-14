import {
  Controller,
  Post,
  Get,
  Body,
  Query,
  UseGuards,
  Headers,
} from '@nestjs/common';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { HealthDataService } from './health-data.service';
import { AiProxyService } from '../ai-proxy/ai-proxy.service';
import { FoodLogDto, FoodHistoryQueryDto, FoodRecognizeDto } from './dto/food.dto';

@Controller('food')
@UseGuards(JwtAuthGuard)
export class FoodController {
  constructor(
    private readonly healthDataService: HealthDataService,
    private readonly aiProxyService: AiProxyService,
  ) {}

  @Post('log')
  async logFood(
    @Body() dto: FoodLogDto,
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    const saved = await this.healthDataService.saveFoodLog(user.userId, dto);

    try {
      await this.aiProxyService.logFood(user.userId, dto, token);
    } catch {
      // AI service unavailable - continue without AI response
    }

    return {
      success: true,
      data: {
        saved: true,
        imageBase64: dto.imageBase64 ? '(image stored)' : undefined,
      },
    };
  }

  @Get('history')
  async getHistory(
    @Query() query: FoodHistoryQueryDto,
    @CurrentUser() user: any,
  ) {
    const history = await this.healthDataService.getFoodHistory(
      user.userId,
      query.from ? new Date(query.from) : undefined,
      query.to ? new Date(query.to) : undefined,
      query.limit,
    );

    return {
      success: true,
      data: history,
    };
  }

  @Post('recognize')
  async recognizeFood(
    @Body() dto: FoodRecognizeDto,
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    // TODO: Add recognizeFood to AiProxyService when needed
    return {
      success: true,
      data: { message: 'Food recognition not yet implemented' },
    };
  }
}

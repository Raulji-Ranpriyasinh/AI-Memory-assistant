import {
  Controller,
  Post,
  Get,
  Body,
  UseGuards,
  Headers,
} from '@nestjs/common';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { AiProxyService } from '../ai-proxy/ai-proxy.service';
import { ChatMessageDto } from './dto/chat.dto';

@Controller('chat')
@UseGuards(JwtAuthGuard)
export class ChatController {
  constructor(private readonly aiProxyService: AiProxyService) {}

  @Post()
  async sendMessage(
    @Body() dto: ChatMessageDto,
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    const response = await this.aiProxyService.chat(user.userId, dto.message, token);

    return {
      success: true,
      data: {
        response: response?.message || response?.response || '',
      },
    };
  }

  @Get('history')
  async getHistory(
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    const history = await this.aiProxyService.getChatHistory(user.userId, token);

    return {
      success: true,
      data: history?.conversations || history?.messages || [],
    };
  }
}

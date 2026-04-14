import { Controller, Post, Body, UseGuards, Get } from '@nestjs/common';
import { AuthService } from './auth.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { TokenResponseDto } from './dto/token-response.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';

@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {
    console.log('[AuthController] AuthController instantiated');
  }

  @Get('test')
  testRoute() {
    return { message: 'Auth controller is working!' };
  }

  @Post('register')
  async register(@Body() dto: RegisterDto): Promise<TokenResponseDto> {
    return this.authService.register(dto);
  }

  @Post('login')
  async login(@Body() dto: LoginDto): Promise<TokenResponseDto> {
    return this.authService.login(dto);
  }

  @Post('refresh')
  @UseGuards(JwtAuthGuard)
  async refresh(
    @Body('token') token: string,
    @CurrentUser() user: any,
  ): Promise<TokenResponseDto> {
    return this.authService.refresh(token);
  }

  @Post('logout')
  @UseGuards(JwtAuthGuard)
  async logout(@CurrentUser() user: any): Promise<{ success: boolean }> {
    return this.authService.logout(user.userId);
  }
}
